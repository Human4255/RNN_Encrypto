    //데이터
    data_sets = []
    //데이터 아키텍처{title:"",content:"",sub_imag:[],user_fill:""}
    data_sets.push({
      sub_title:"이미지크롤링",
      sub_content:"구글사이트 및 네이버사이트를 이동하며 해당 검색어의 이미지를 수집",
      sub_img:["1BfprTRpF9pYhtvVQulFp4tFpqHoKxvaG","1elWDCkAc4-7BfrKh0uHk_SKRgQ-BEOGs","1ugILTJhYHZ2srmnxiONw7Es-7PJ9208s"],
      user_fill:"opencv,selenium라이브러리와 크롬드라이버를 활용해 데이터 수집 자동화",
      asso_file:"<관련파일> /Crawling/image_crawling_running 실행"
    })
    //이미지전처리
      data_sets.push({
      sub_title:"이미지증대 및 배경제거",
      sub_content:"수집된 이미지의 밝기,회전 등을 조정하여 수량을 증대시키고 학습의 효율을 높이기 위해 이미지 배경을 제거함",
      sub_img:["1s1ti4noMAaXLeAkdm3JEvlsrYVsM983-","1uQ3LxCYx2JMKs139V7P_T_Kq3MvTzPXO","1g3pzjNaZxBKroKvWYDGNqbdkqY3Ga43H"],
      user_fill:"배경제거를 위해 rembg 라이브러리와 증대를 위해 텐서플로우의 밝기,회전 등의 조정 라이브러리를 사용",
      asso_file:"<관련파일> /Preprcessing/preprocessing_running 호출 실행"
    })
      data_sets.push({
      sub_title:"이미지전처리",
      sub_content:"이미지의 사이즈 및 수치 표준화 및 정답파일의 원핫인코딩 후 훈련데이터셋 준비",
      sub_img:["1g2iS_CdlYJxUbmP-NPHPtRUP4QC6swu_","1-d-m3TEfjAdR5CxKic2FDlYv1UInol1l"],
      user_fill:`이미지의 사이즈를 64*64로 조정하였으며, 색상의 최댓값 255를 이용하여 0~1 사이의 수치로 min-max scale를 실행,
                 정답데이터를 원핫인코딩을 이진데이터로 변경수행`,
      asso_file:"<관련파일> /Preprcessing/preprocessing_running, Trainning/construct_model 호출 실행"
    })
      data_sets.push({
      sub_title:"데이터 확인 및 모델구성",
      sub_content:"전처리가 마무리 되고 정답과 이미지의 순서별 일치를 확인 후 모델구성 훈련실행",
      sub_img:["1jzuDQqrNej1_Wg8qLK5ewH7NtoMBtLMO","114fjGf2IvaMR4PW0koRTkoveAK0f0zrg","1zeL_gMadJ6X4gv3JJ9RCoLV4vTZkUgGm",],
      user_fill:"컨볼루션 레이어와 맥스풀 및 드랍아웃 레이어, learnning_rate를 이용해 특성 추출과 과적합 방지",
      asso_file:"<관련파일> /Trainning/construct_model 일부 호출 실행"
    })
      data_sets.push({
      sub_title:"훈련시작과 종료",
      sub_content:"구성된 모델과 전처리된 데이터를 이용해 모델을 훈련",
      sub_img:["1RYrvEm0DBERnyNNPba2nqClbVNhWV1_K","14kY-8uuIFZ-KJWCRUZWACWx0vBWJuJrW"],
      user_fill:"최적의 검증데이터 정확도를 찾기위해 조기종료 콜백함수와 adam의 최적화 함수, Categorical_crossentropy손실함수사용",
      asso_file:"<관련파일> /Trainning/construct_model 호출 실행"
    })
      data_sets.push({
      sub_title:"훈련된 모델의 평가 및 예측",
      sub_content:"훈련이 완료된 모델 결과의 시각화",
      sub_img:["1SbuOivBqmp6wRWB7dFK0KPCw27K1v83Q","1oauqBKdgZ90gXqsO0JNyjN_ggZy_xpZZ"],
      user_fill:"훈련횟수가 부족하고, 이미지 다양성 부족으로 인한 과대적합 증상이 보임 더 많은 이미지가 필요함",
      asso_file:"<관련파일> /Trainning/construct_model 호출 실행"
    })
      data_sets.push({
      sub_title:"모델의 혼동행렬 시각화 및 f1 score 요약",
      sub_content:"모델의 종합 성능평가와 개선을 위해 혼동행렬 표기",
      sub_img:["12f6jLaInHYsVt3zAb1_e4Z3U3_mRsgIv","13ReYyIc4pzyL1sp57uWt8eLCIMdwBj8e","14HVojDEGXg_8MGxwYGreY2wsx1ykawHd"],
      user_fill:"이미지 다양성 부족으로 인한 과대적합 증상이 보임 더 많은 이미지가 필요함",
      asso_file:"<관련파일> /Trainning/construct_model 호출 실행"
    })
      data_sets.push({
      sub_title:"실제 데이터 예측 테스트",
      sub_content:"새로운 이미지를 구비하여 자동 전처리와 모델에 예측구현",
      sub_img:["1kdeUXDTQy9ul03CetNoAmAdJaxvxpY0s","1Uf7gGQWVoxLNhK5SjBkQU5UjMXbrhTwB","1RHYd2KObYHQKzNyI_nuQxKYjfft_OAee","11TTXW107cAXwHQFzXN2N0AawhBm-xr9U"],
      user_fill:"이미지 다양성의 부족으로 성능은 다소 떨어지지만 특징이 뚜렷한 이미지들은 예측가능",
      asso_file:"<관련파일> /SampleData_predict/test_class 호출 실행"
    })