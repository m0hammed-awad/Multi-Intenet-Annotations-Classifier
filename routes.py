import os
import logging
import requests
import json
import re
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AIService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

        # If you still want hardcoded key (NOT recommended), uncomment:
        # self.api_key = "YOUR_API_KEY_HERE"

        if not self.api_key:
            logger.error("Gemini API key missing. Set GEMINI_API_KEY.")
            self.client_available = False
        else:
            self.client_available = True
            logger.info("Gemini FAST client initialized")

        self.model_name = "gemini-2.0-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.session = requests.Session()

        self.labels1 = [
            'cancel_order', 'change_order', 'change_shipping_address', 'check_cancellation_fee',
            'check_invoice', 'check_payment_methods', 'check_refund_policy', 'complaint',
            'contact_customer_service', 'contact_human_agent', 'create_account', 'delete_account',
            'delivery_options', 'delivery_period', 'edit_account', 'get_invoice', 'get_refund',
            'newsletter_subscription', 'payment_issue', 'place_order', 'recover_password',
            'registration_problems', 'review', 'set_up_shipping_address', 'switch_account',
            'track_order', 'track_refund'
        ]

        self.labels2 = [
            'ACCOUNT', 'CANCELLATION_FEE', 'CONTACT', 'DELIVERY', 'FEEDBACK',
            'INVOICE', 'NEWSLETTER', 'ORDER', 'PAYMENT', 'REFUND', 'SHIPPING_ADDRESS'
        ]

        self.system_prompt = """
You are a professional customer support chatbot for an e-commerce service.
Reply short, helpful, and polite.
Never say you are an AI model.
Ask only 1 question if needed.
"""

    # -----------------------------
    # GEMINI CALL WITH RETRY (429 FIX)
    # -----------------------------
    def _gemini_generate(self, prompt, max_tokens=250, temperature=0.2, retries=3):
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(retries):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=15)

                # If rate limited -> wait and retry
                if response.status_code == 429:
                    wait_time = (2 ** attempt)  # 1s, 2s, 4s
                    logger.error(f"429 Rate Limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return ""

                parts_out = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in parts_out]).strip()

            except Exception as e:
                logger.error(f"Gemini call failed (attempt {attempt+1}): {e}")
                time.sleep(1)

        return ""

    # -----------------------------
    # SINGLE CALL: INTENT + RESPONSE
    # -----------------------------
    def generate_response(self, user_message, conversation_history=None):
        if not self.client_available:
            return self._fallback_response(), "contact_customer_service (50%)", "CONTACT (50%)"

        # Keep history minimal for speed
        context = ""
        if conversation_history:
            recent = conversation_history[-2:]
            for m in recent:
                role = "Customer" if m.get("role") == "user" else "Agent"
                context += f"{role}: {m.get('content','')}\n"

        prompt = f"""
{self.system_prompt}

You must do TWO tasks:
Task-1: classify intent from labels1 and labels2
Task-2: generate support reply

labels1: {self.labels1}
labels2: {self.labels2}

Return JSON only in this exact format:
{{
  "reply":"...",
  "intent1":"<one from labels1>",
  "conf1":0.xx,
  "intent2":"<one from labels2>",
  "conf2":0.xx
}}

Conversation:
{context}

Customer message:
{user_message}
""".strip()

        out = self._gemini_generate(prompt, max_tokens=280, temperature=0.2, retries=3)

        try:
            match = re.search(r"\{.*\}", out, re.DOTALL)
            if not match:
                return self._fallback_response(), "contact_customer_service (50%)", "CONTACT (50%)"

            obj = json.loads(match.group())

            reply = obj.get("reply", "").strip()
            intent1 = obj.get("intent1", "contact_customer_service")
            intent2 = obj.get("intent2", "CONTACT")
            conf1 = float(obj.get("conf1", 0.50))
            conf2 = float(obj.get("conf2", 0.50))

            # strict validation
            if intent1 not in self.labels1:
                intent1, conf1 = "contact_customer_service", 0.50
            if intent2 not in self.labels2:
                intent2, conf2 = "CONTACT", 0.50

            conf1 = max(0.0, min(conf1, 1.0))
            conf2 = max(0.0, min(conf2, 1.0))

            intent1_display = f"{intent1} ({int(conf1 * 100)}%)"
            intent2_display = f"{intent2} ({int(conf2 * 100)}%)"

            if not reply:
                reply = self._fallback_response()

            return reply, intent1_display, intent2_display

        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return self._fallback_response(), "contact_customer_service (50%)", "CONTACT (50%)"

    def _fallback_response(self):
        return "Thanks for contacting support. Please share your Order ID or registered email so I can assist you."

    def is_available(self):
        return self.client_available
