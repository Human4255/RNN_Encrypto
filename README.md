::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: describe :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    

시계열데이터 회귀분석예측을 위해 가상화폐 API를 이용하여 향후 가격을 예측한다.  
지속적인 훈련을 통해 모델의 성능이 향상될 수 있다.
    
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

기본라이브러리: numpy, tensorflow, matplotlib.pyplot, os, random, pickle 

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
1. 크롤링작업    
 Crawling/Craw_image_running :: 실행파일    
 Crawling/image_crawlling_google :: 구글크롤링함수 
 Crawling/image_crawlling_naver :: 네이버크롤링함수 
    이미지 저장경로, 검색어, 영문검색, 다운로드 개수    
    크롬 드라이버 필요 (https://googlechromelabs.github.io/chrome-for-testing/) Version: 131.0.6778.108  
    크롬 웹브라우저 필요 Version: 131.0.6778.108 (x64)
    selenium 라이브러리 필요
    
2. 데이터 증강 및 배경제거
Preprocessing/preprocessing_running :: main run
Preprocessing/preprocessing_Background :: import
Preprocessing/Utill :: import
    이미지 배경 제거 및 무작위로 밝기, 회전, 상하좌우반전, 확대/축소 시키기
        :::실습환경:::
      - rembg.remove
      - tf.image.random_brightness
      - tf.keras.layers.RandomRotation
      - tf.keras.layers.RandomFlip
      - tf.keras.layers.RandomZoom

3. 데이터 전처리 및 훈련실행, 평가
Trainning/train_fit :: main run
Trainning/construct_model :: import 
    이미지 전처리 및 훈련실행과 훈련결과를 평가함
    훈련 실행 시 최적값으로 조기종료 콜백을 등록함
        :::실습환경:::
      - from tensorflow.keras import Sequential, Input
      - from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
      - from sklearn.metrics import confusion_matrix, classification_report
      - import seaborn
   *에러처리는 아직 하지 않음 미완성

4. 실제 이미지 처리 샘플 확인
SampleData_predict/test_class :: main run
    인터넷에서 가져온 이미지 파일을 모델이 측정
        :::준비물:::
      - 단일이미지

* 순서대로 실행 시 생성되는 파일리스트
* Trainning/classification_image.keras(저장모델)
* classcification_image.history(훈련결과 손실수치)
* config(영문 라벨리스트)
* eroor처리 수행하지 않음
* 원본 이미지 60*10개, 증강이미지 60*10개

copy right - 2024.12.18 광주컴퓨터기술학원 (이선하) 

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  

r"D:\imgs": 디렉터리 내의 폴더명은 정답으로 처리, 각 관련 이미지가 내부에 존재해야 한다.           

opencv0.y: def RemoveBackground 디렉터리 내의 모든 이미지 백그라운드 제거 및 이미지 256*256으로 리사이징           
           def SingleRemoveBackground 단일 이미지에 대한 256*256으로 리사이징           

classification_model.py : def imageAugment_sub 디렉터리 내의 이미지를 무작위로 밝기, 회전, 상하좌우 반전, 확대/축소 시켜 이미지를 증강시킴                      
                          def ReadImage 디렉터리 내의 이미지를 불러와 증강함수(imageAugment_sub)를 호출하여 이미지를 저장함           
                          def Load_directory_sub 증강된 이미지를 읽어 {label:[이미지리스트]}형태로 변환, 64*64으로 리사이징, 정답리스트, y_label리스트로 저장           
                          def GetTrainData Load_directory_sub함수를 호출하여 정답파일을 기준으로 훈련데이터(80%), 테스트데이터(20%)를 랜덤하게 분리           
                       
construct_model.py : 훈련데이터의 minmax표준화, 정답데이터의 원핫인코딩을 수행후, 이미지와 정답파일의 매칭을 확인           
                     Sequential모델로 구성된 CNN을 구현하고(다중분류) 훈련을 실행+조기종료 콜백함수 추가 후 훈련결과와 모델을 파일로 저장           
                    +) max-acc-baseline기준이상, min-loss-baseline기준이하           

classfication_analize.py : 테스트데이터와 테스트정답파일을 로딩한 후 훈련이 완료된 모델과 훈련결과를 로딩하여           
                           훈련모델의 예측값을 출력해 실제 정답과 이미지와 예측정답을 시각화표현 -matplotlib.pyplot사용           
                           혼돈행렬 진행 후 히트맵을 그려줌  
