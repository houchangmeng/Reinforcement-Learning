import numpy as np
from model import QModel
import tensorflow as tf
import copy

checkpoint_save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/a3_DuelDQN/checkpoint/dqn_cartpole.ckpt'

class Agent(object):
    '''
    Duel DQN
    '''
    def __init__(self,e_greedy = 0.1,e_greedy_decay =0.001,e_greedy_min = 0.01) -> None:
        self.e_greedy = e_greedy
        self.e_greedy_decay =  e_greedy_decay
        self.e_greedy_min =  e_greedy_min
        self.learn_num = 0
        self.action_space = [0,1]

        self.model = QModel()
        self.target_model = copy.deepcopy(self.model)
        self.lr = 1e-3
        self.gamma = 0.99

        self.model_optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr)

    def predict(self,observation):
        if (np.random.uniform() > self.e_greedy) : 
            action=np.random.choice(self.action_space)
        else:
            observation = np.array([observation])
            q = self.model(observation).numpy()
            action = np.argmax(q[0])
        return action
        
    def learn(self,transtion_batch):

        observations = tf.convert_to_tensor(transtion_batch[0])
        actions = tf.convert_to_tensor(transtion_batch[1],dtype='int32')
        actions = tf.expand_dims(actions,axis=1)
        rewards = tf.convert_to_tensor(transtion_batch[2])
        rewards = tf.expand_dims(rewards,axis=1)
        next_observations = tf.convert_to_tensor(transtion_batch[3])
        dones = tf.convert_to_tensor(transtion_batch[4])
        dones = tf.expand_dims(dones,axis=1)

        with tf.GradientTape() as tape:
            target_q = tf.reduce_max(self.target_model(next_observations),axis=1)
            target_q = tf.expand_dims(target_q,axis=1)
            target_q = rewards + (1-dones)*self.gamma * target_q

            q_predict = self.model(observations)
            q_target = q_predict.numpy()
            batch_size = len(dones)
            row = tf.expand_dims(tf.range(batch_size,dtype=tf.int32),axis=1)
            col = actions
            q_target[row,col] = target_q
            loss = tf.losses.mse(q_target,q_predict)
        grads = tape.gradient(
            loss,self.model.trainable_variables)
        self.model_optimizer.apply_gradients(
            zip(grads,self.model.trainable_variables))
        
        new_greedy = 0.1 + 0.89 * (1 - np.exp(-self.learn_num/100))
        if self.e_greedy < new_greedy:
            self.e_greedy = new_greedy

        self.learn_num+=1
        if self.learn_num % 100 == 0:
            self.sync_model()

    def sync_model(self):
        # sync target model and model
        self.target_model.set_weights(self.model.get_weights())

    def save_models(self):
        self.model.save_weights(checkpoint_save_path)

    def load_models(self):
        self.model.load_weights(checkpoint_save_path)
            


