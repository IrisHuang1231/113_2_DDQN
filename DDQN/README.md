0. 下載所需套件

pip install -r requirements.txt

檢查已安裝套件

pip list

1. 訓練模型

python run.py --mode train

更改參數

python run.py --mode train -e 10

2. 測試模型

python run.py --mode test --weights ./weights/202504301430-dqn.weights.h5 -e 500


*DQN
episode: 500/500, episode end value: 8670.453584822495
mean portfolio_val: 8644.295381133572
median portfolio_val: 8548.27748720002

*DDQN
episode: 500/500, episode end value: 8351.513292237465
mean portfolio_val: 8660.90194606553
median portfolio_val: 8629.829449562496

*換景氣循環
episode: 500/500, episode end value: 8927.214324595006
mean portfolio_val: 8385.781456189505
median portfolio_val: 8341.964648576228

*修正動作與回報
episode: 500/500, episode end value: 24256.808205462552
mean portfolio_val: 24999.85574295905
median portfolio_val: 25071.944337981164

*改總經
episode: 500/500, episode end value: 25980.860184060966
mean portfolio_val: 25651.853951080688
median portfolio_val: 25739.83135186039
