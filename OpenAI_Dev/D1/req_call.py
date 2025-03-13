# REST
import http.client
import json

conn = http.client.HTTPSConnection("api.openai.com")
payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
})
headers = {
    'Accept': 'application/json',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ...',
    'Host': 'api.openai.com',
    'Connection': 'keep-alive'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))