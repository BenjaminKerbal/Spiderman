# Spiderman

<h3>
When I was a kid, I played a Spiderman game that was available as a browser game. The goal was to survive as long as possible by swinging forward. After my studies were completed, I finally found time to create a replica of the game according to my liking. While developing the game I started to think about implementing a reinforcement learning agent to play the game and master it. That is where this project really got interesting. 
</h3>

<h3>
During my studies I have taken many machine learning courses, such as deep learning and reinforcement learning, and I wanted to apply those methods to this project. Currently, the only available agent is a Double Deep Q Network (DDQN), which uses a Recurrent Neural Network (RNN) structure to remember what it has seen in order to select the correct actions. To speed up the training process, I implemented multi-threading possibilities which allowed 32 environments to be played simultaneously. The network update is accelerated using GPU to further increase the training process speed. The pretrained model has trained approximately 1 million timesteps to achieve its performance. While the agent could be improved, I am quite satisfied with the current results.
</h3>

![](Extra/ExampleGame.gif)




