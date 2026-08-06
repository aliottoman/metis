# langchain-oci: vision extraction and DAC switching

Verified against the installed package `langchain-oci==0.3.1` (source
inspected, not recalled). Applies to chat models on OCI Generative AI,
including Cohere Command A Vision.

## The one client, both serving modes

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id=MODEL_ID,                      # see the switching rule below
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_OCID,
    provider="cohere",                      # REQUIRED for a DAC OCID; derived from
                                            # the model name only in on-demand mode
    auth_type="API_KEY",                    # default; reads ~/.oci/config
    auth_profile="DEFAULT",
    model_kwargs={"temperature": 0, "max_tokens": 2048},
)
```

**On-demand ↔ Dedicated AI Cluster is ONE string.** Verified in the source:
a `model_id` that starts with `ocid1.generativeaiendpoint` is sent as
`DedicatedServingMode(endpoint_id=…)`; anything else is
`OnDemandServingMode(model_id=…)`. So the alternation the app needs is an
environment variable, not a second client:

```python
# .env / config — the app reads ONE name and the serving mode follows:
#   on-demand:  OCI_VISION_MODEL=cohere.command-a-vision-07-2025
#   DAC:        OCI_VISION_MODEL=ocid1.generativeaiendpoint.oc1.us-chicago-1.amaaa…
MODEL_ID = os.environ["OCI_VISION_MODEL"]
```

When the value is a DAC endpoint OCID, `provider="cohere"` must be passed
explicitly — the provider cannot be derived from an OCID and the client
raises without it.

## Sending a document image for extraction

`HumanMessage.content` is a LIST of typed parts. The package ships the
encoder; do not hand-build data URIs:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_oci import encode_image   # bytes -> content block
# also available: load_image(path) for files on disk

message = HumanMessage(content=[
    {"type": "text", "text": "Extract supplier, amount, date and IBAN from this invoice."},
    encode_image(image_bytes, mime_type="image/png"),
])
reply = llm.invoke([SystemMessage(content=SYSTEM), message])
text = reply.content
```

`encode_image` returns `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`.
Cohere vision models are routed over OCI's V2 chat API automatically when
multimodal content is detected — nothing to configure.

Vision-capable model families (from the package's own registry):
`cohere.command-a-vision`, `meta.llama-3.2-11b/90b-vision-instruct`,
`meta.llama-4-scout/maverick`, `xai.grok-4` family, `google.gemini-2.5`
family. The dated on-demand name for Cohere is
`cohere.command-a-vision-07-2025`.

## Structured extraction

`ChatOCIGenAI` is a standard LangChain chat model:
`llm.with_structured_output(MyPydanticModel)` returns the parsed model.
For extraction pipelines prefer a Pydantic contract over free text — the
downstream rule checks then read fields, not prose.

## Mistakes this document exists to prevent

- Building a second client class for DAC. Wrong: same class, OCID model_id.
- Passing the DAC OCID as `service_endpoint`. Wrong: the service endpoint
  stays the regional inference URL; the OCID goes in `model_id`.
- Hand-rolling `{"role": "user", "content": "<base64>"}`. Wrong shape: use
  content parts with `encode_image`/`load_image`.
- Reading config at import time. Read env inside the request handler or a
  lazy accessor, so the app imports and serves with no environment (the
  verifier enforces this).
