import openai

class LLM_Callable:
    def __init__(self,base_url,api_key,model_name):
        self.base_url=base_url
        self.api_key=api_key
        self.model_name=model_name
        self.client=openai.Client(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
    def invoke(self,inputs):
        try:
            response=self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role":"user","content":inputs}
                ]
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            # Do not swallow connection/proxy/model errors.
            # Callers decide fallback strategy and can expose precise error cause.
            raise RuntimeError(f"LLM invoke failed: {type(e).__name__}: {e}") from e

