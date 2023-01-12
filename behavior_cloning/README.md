# Behavior Cloning

1. SAC 训练一个比较好的Agent(tensorflow)  
    agent.py  
    model.py  
    replay_buffer.py  
    train.py  
    play.py
2. 用训练好的Agent文件收集数据集(pytorch)  
    collect_data.py
3. 训练新的policy
   beavior_cloning.py

一般情况下通过模仿学习得到policy和强化学习的agent效果差不多，但有时候模仿学习policy会冲出跑道，强化学习Agent未发现这种情况。
