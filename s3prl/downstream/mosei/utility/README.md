Notes:  
convert_label.sh is used to generate CMU_MOSEI_Labels.csv.
You may need to change permission of that bash file to run.
```
chmod +x convert_label.sh
./convert_label.sh
```

You will get the messsage below if generation of csv file success.
```
Info of Dataset
---------- Original Video -----------                                  
Num of videos in training   set:  2249                                          
Num of videos in validation set:  300                                           
Num of videos in testing    set:  678
-------------------------------------
---------- Segmented Audio ----------
Num of Training   Data:  16327
Num of Validation Data:  1871
Num of Testing    Data:  4662
Num of Useless    Data:  399
Num of Useful     Data:  22860
Total             Data:  23259
--------------------------------------
---------- 2-class sentiment ---------
Class non-negative (1):  16576
Class negative     (0):  6683
--------------------------------------
---------- 3-class sentiment ---------
Class positive (1):  11476
Class neutral  (0):  5100
Class negative (-1):  6683
--------------------------------------
--------- 6-class emotion ------------
Class happiness (1):  14567
Class sadness   (2):  3782
Class anger     (3):  2730
Class surprise  (4):  437
Class disgust   (5):  1291
Class fear      (6):  452
--------------------------------------
--------- 7-class sentiment ----------
Class -3:  821
Class -2:  2253
Class -1:  3609
Class +0:  5100
Class +1:  7576
Class +2:  3225
Class +3:  675
```