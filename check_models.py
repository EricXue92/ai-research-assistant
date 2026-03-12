import anthropic
client = anthropic.Anthropic()
models = client.models.list()
for m in models.data:
    print(m.id)
