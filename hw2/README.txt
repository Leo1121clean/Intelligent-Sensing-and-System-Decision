#for training
1. 執行data_generator.py生成資料集
2. 執行make_odgt.py生成路徑檔
3. 執行train.py訓練模型

#for reconstructing
1. 執行load.py，使用w,a,d按鍵蒐集重建圖片資料集(包括annotation)
2. 執行make_odgt.py生成路徑檔
3. 執行eval_multipro.py生成語義分割後之圖片
4. 執行3d_semantic_map.py進行重建