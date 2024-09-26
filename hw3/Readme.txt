1.執行map.py生成一張2D地圖
2.執行navigation.py: (navigation_birrt.py為Bi-RRT演算法版本)
  (1)輸入欲抵達的物件
  (2)在地圖上雙擊左鍵設定出發點，接著會生成規劃完之路徑並顯示
  (3)按任意鍵後，即會開始顯示導航動畫

#若欲單純測試RRT演算法(不導航)，可分別單獨執行以下：
  (1)rrt.py (初始RRT)
  (2)rrt_smooth.py (Smooth過的RRT)
  (3)birrt.py (Bi-RRT)