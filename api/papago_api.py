import os
import sys
import requests
from pprint import pprint

# client_id = "oqxI18HsccN_Np2JQfyf" # 개발자센터에서 발급받은 Client ID 값
# client_secret = "deZQTDorha" # 개발자센터에서 발급받은 Client Secret 값
# # client_id = "v6Z86kiMQclwjL_ZotuZ" # 개발자센터에서 발급받은 Client ID 값
# # client_secret = "UcGhi_8gJG" # 개발자센터에서 발급받은 Client Secret 값
client=[("oqxI18HsccN_Np2JQfyf","deZQTDorha"),("QjYEOdbFXJKrBlPSfyAy","AxQrHPbUxm"),("v6Z86kiMQclwjL_ZotuZ","UcGhi_8gJG")]
def get_translate(text):
    data = {'text' : text,
            'source' : 'en',
            'target': 'ko'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    for client_id,client_secret in client:
        header = {"X-Naver-Client-Id":client_id,
                  "X-Naver-Client-Secret":client_secret}

        response = requests.post(url, headers=header, data= data)
        rescode = response.status_code

        if(rescode==200):
            t_data = response.json()
            #pprint(t_data['message']['result']['translatedText'])
            return t_data['message']['result']['translatedText']
        else:
            print("Error Code:" , rescode)
    return -1