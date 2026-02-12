import os
import logging
import base64
import requests
import mimetypes
import json
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AIService:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            logger.error("OpenRouter API key missing.")
            self.client_available = False
        else:
            self.client_available = True
            logger.info("OpenRouter client initialized")

        # Default model from OpenRouter - using Arcee Trinity as requested
        self.model_name = "arcee-ai/trinity-large-preview:free"

        # reuse connection (faster)
        self.session = requests.Session()

        # Labels for intent classification (aligned with routes.py)
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

        # Support system prompt
        self.system_prompt = """
You are a professional customer support chatbot for an e-commerce service.
Reply short, helpful, and polite.
Never say you are an AI model.
Ask only 1 question if needed.
""".strip()

    # ---------------------------
    # OPENROUTER CALL (REUSABLE)
    # ---------------------------
    def _openrouter_generate(self, messages, max_tokens=250, temperature=0.2, enable_reasoning=True, model_override=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Filter messages to ensure only valid fields are sent to OpenRouter
        clean_messages = []
        for m in messages:
            clean_msg = {"role": m["role"], "content": m["content"]}
            # Preserve reasoning_details for models that support it (continued reasoning)
            if "reasoning_details" in m:
                clean_msg["reasoning_details"] = m["reasoning_details"]
            clean_messages.append(clean_msg)

        payload = {
            "model": model_override if model_override else self.model_name,
            "messages": clean_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_reasoning:
            payload["reasoning"] = {"enabled": True}
        
        logger.debug(f"OpenRouter Payload: {json.dumps(payload)}")

        try:
            response = self.session.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=30)
            logger.info(f"OpenRouter response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"OpenRouter response data: {json.dumps(data)}")
            
            choice = data['choices'][0]['message']
            content = choice.get('content') or ""
            reasoning_details = choice.get('reasoning_details')
            
            # Special case: some models return everything in 'reasoning'
            if not content and 'reasoning' in choice:
                content = choice['reasoning']
            
            return content, reasoning_details
        except Exception as e:
            logger.error(f"OpenRouter call failed: {e}")
            return "", None

    # ---------------------------
    # INTENT DETECTOR (SINGLE WORD + %)
    # ---------------------------
    def detect_intent(self, user_message):
        """
        Returns dict like:
        {"intent1": "payment_issue", "intent2": "PAYMENT", "confidence1": 0.92, "confidence2": 0.95}
        """

        prompt = f"""
Classify the customer message into exactly ONE intent from labels1 and exactly ONE from labels2.

labels1: {self.labels1}
labels2: {self.labels2}

Return JSON only in this exact format:
{{"intent1":"<one_from_labels1>","conf1":0.xx,"intent2":"<one_from_labels2>","conf2":0.xx}}

Message: {user_message}
""".strip()

        try:
            # Stricter prompt for better JSON compliance
            messages = [{"role": "user", "content": f"TASK: Classify intent into JSON.\n\n{prompt}\n\nIMPORTANT: ONLY JSON. NO TEXT."}]
            
            # Use Arcee Trinity for classification as well
            out, _ = self._openrouter_generate(
                messages, 
                max_tokens=600, 
                temperature=0.0, 
                enable_reasoning=False,
                model_override="arcee-ai/trinity-large-preview:free"
            )
            logger.debug(f"Raw intent output: {out}")
            match = re.search(r"\{.*\}", out, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                intent1 = str(obj.get("intent1", "contact_customer_service")).lower().strip()
                intent2 = str(obj.get("intent2", "CONTACT")).upper().strip()
                conf1 = float(obj.get("conf1", 0.50))
                conf2 = float(obj.get("conf2", 0.50))

                # enforce allowed intents
                if intent1 not in self.labels1:
                    intent1 = "contact_customer_service"
                if intent2 not in self.labels2:
                    intent2 = "CONTACT"

                conf1 = max(0.0, min(conf1, 1.0))
                conf2 = max(0.0, min(conf2, 1.0))
                return {
                    "intent1": intent1, 
                    "confidence1": conf1,
                    "intent2": intent2,
                    "confidence2": conf2
                }

        except Exception as e:
            logger.error(f"Intent detection error: {e}")

        return {
            "intent1": "contact_customer_service", 
            "confidence1": 0.50,
            "intent2": "CONTACT",
            "confidence2": 0.50
        }

    # ---------------------------
    # MAIN CHAT RESPONSE
    # ---------------------------
    def generate_response(self, user_message, conversation_history=None):
        logger.info(f"Generating response for message: {user_message}")
        if not self.client_available:
            return self._fallback_response(user_message), {"intent1": "contact_customer_service", "intent2": "CONTACT", "confidence1": 0.5, "confidence2": 0.5}, None

        # detect intent
        intent_obj = self.detect_intent(user_message)
        logger.info(f"Detected intent object: {intent_obj}")
        
        intent1 = intent_obj["intent1"]
        conf1 = intent_obj["confidence1"]
        intent2 = intent_obj["intent2"]
        conf2 = intent_obj["confidence2"]
        
        intent1_display = f"{intent1} ({int(conf1*100)}%)"
        intent2_display = f"{intent2} ({int(conf2*100)}%)"

        # Prepare messages for OpenRouter
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if conversation_history:
            for msg in conversation_history:
                role = msg.get('role')
                content = msg.get('content')
                reasoning = msg.get('reasoning_details')
                
                msg_obj = {"role": role, "content": content}
                if reasoning:
                    msg_obj["reasoning_details"] = reasoning
                messages.append(msg_obj)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            # For general e-commerce queries, enable reasoning
            enable_r = True
            # For very short greetings, maybe disable to save time/tokens? 
            # But let's keep it on for now as requested.
            
            reply, reasoning_details = self._openrouter_generate(messages, enable_reasoning=enable_r)

            if reply:
                return reply, intent_obj, reasoning_details

        except Exception as e:
            logger.error(f"Response generation error: {e}")

        return self._fallback_response(user_message), intent_obj, None

    def _fallback_response(self, user_message):
        return "Please tell me your issue—delivery, payment, refund, account, or feedback."

    def is_available(self):
        return self.client_available
