import requests
import json

url = "http://localhost:5001/invocations"  # Update with your model's URL
data = {
  "instances": [
    [
      0.00632,   # CRIM
      18.0,      # ZN
      2.31,      # INDUS
      0,         # CHAS
      0.538,     # NOX
      6.575,     # RM
      65.2,      # AGE
      4.09,      # DIS
      1,         # RAD
      296.0,     # TAX
      15.3,      # PTRATIO
      396.9,     # B
      4.98       # LSTAT
    ],
    [
      0.00132,  # CRIM
      18.0,      # ZN
      2.31,      # INDUS
      5,        # CHAS
      0.538,     # NOX
      6.575,     # RM
      65.2,      # AGE
      4.09,      # DIS
      1,         # RAD
      12.0,     # TAX
      15.3,      # PTRATIO
      396.9,     # B
      4.98       # LSTAT
    ]
  ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
