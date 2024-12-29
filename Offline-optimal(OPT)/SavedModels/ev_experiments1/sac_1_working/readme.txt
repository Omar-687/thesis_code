much bigger o2 parameter that relates to penalties for not charging enough causes a lot of problems
this o2 parameter seems to be better
try lowering it a bit

This is a shaped reward, meaning it gives increasing reward in states that are closer
 to the end goal. This is in contrast to sparse rewards, which give reward at the goal state, and no reward anywhere else. Shaped rewards are often much easier to learn, because they provide positive feedback
even when the policy hasn’t figured out a full solution to the problem.

if the rewards are sparse the agent will have to explore more to get rewards and learn the optimal policy, whereas if the rewards are dense in time, the agent is quickly guided towards its learning goal

 Essentially, sparser rewards make for a harder problem to solve. All RL algorithms can cope with sparse rewards to some degree, the whole concept of returns and value backup is designed to deal with sparseness at a theoretical level. In practical terms however, it may take some algorithms an unreasonable amount of time to determine a good policy beyond certain levels of sparseness.

 smaller gamma also causes problems

 even the same code run 2x can produce different results due to randomness