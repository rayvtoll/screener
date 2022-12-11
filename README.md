# screener
Creates a list of 'interesting pairs' of a desired Exchange on a desired timeframe based on TA indicators

## instructions
1. pip install -r requirements.py
2. create .env file in same directory as screener.py, with variabeles like:
- KUCOIN_REGULAR_API_KEY='your_api_key'
- KUCOIN_REGULAR_API_SECRET='your_api_secret'
- KUCOIN_REGULAR_API_PASSPHRASE='your_api_passphrase'
3. python -W ignore screener.py
