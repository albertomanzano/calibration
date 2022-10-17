import tensorflow as tf
import numpy as np


class Model:

    def __init__(self,drift,diffusion):
        self.drift = drift
        self.diffusion = diffusion

    @tf.function
    def euler_maruyama(self,t,S,dt,dW):
            t_array = tf.ones_like(S)*t
            return S+self.drift(t_array,S)*dt+self.diffusion(t_array,S)*tf.math.sqrt(dt)*dW

    @tf.function
    def MC_paths(self,S0,T,n_paths = 100000,n_steps = 30):
        t = tf.linspace(0.,T[T.shape[0]-1],n_steps) 
        t = tf.concat((t,T), axis = 0)
        order = tf.argsort(t)
        t = tf.gather(t, order)
        index = tf.reshape(tf.where(order>n_steps-1),[-1])
        
        dW = tf.constant(np.random.randn(n_paths,t.shape[0]), dtype = tf.float32) 
        S = S0*tf.ones(n_paths)
        
        paths = []
        paths.append(S)
        for i in range(t.shape[0]-1):
            dt = t[i+1]-t[i]
            S = self.euler_maruyama(t[i],S,dt,dW[:,i])
            paths.append(S)
        
        return t, tf.stack(paths, axis = 1), index

    @tf.function
    def plain_vanilla(self,S0,T,K,discounting = 0.0,n_paths = 100000,n_steps = 100):
        t, paths, index = self.MC_paths(S0,T,n_paths,n_steps)
        final_paths = tf.gather(paths,indices = index,axis = 1)

        K = tf.reshape(K,(-1,1))
        prices = []
        for i in range(len(T)):
            payoff = tf.math.maximum(final_paths[:,i]-K,0.0)
            prices.append(tf.exp(-discounting*T[i])*tf.reduce_mean(payoff, axis = 1))
        return tf.stack(prices,axis = 1)
    
    def plain_vanilla_loss(self,S0,T,K,discounting,market_prices,n_paths = 200000,n_steps = 100):
        predicted_prices = self.plain_vanilla(S0,T,K,discounting,n_paths,n_steps)
        l2 = tf.math.square(market_prices-predicted_prices)
        loss = tf.reduce_sum(l2)
        return loss


