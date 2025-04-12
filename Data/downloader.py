import gdown

url = 'https://drive.google.com/uc?id=1P5viA8hMm19n-Ia3k9wZyQTEloCk2gMJ'
output = 'SSN.zip'

gdown.download(url=url, output=output, fuzzy=True)
