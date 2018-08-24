## TODOs

* Profile memory
* Note that preprocessing is the hardest thing to get right
* check tensorflow version on Euler
* test pipenv on Euler
* Clean up runs repo
* TODO: how much faster? How much slower than A2C?

# TensorFlow A3C

TensorFlow implementation of [A3C](https://arxiv.org/abs/1602.01783) for Atari games using OpenAI
Gym, crafted with love, kindness - and since deep RL is
[apparently hard to get right](https://blog.openai.com/openai-baselines-dqn/),
lots and lots of testing.


## Usage

### Python setup

To set up an isolated environment and install dependencies, install
[Pipenv](https://github.com/pypa/pipenv) (e.g. `pip install --user pipenv`), then just run:

`$ pipenv install`

However, note that TensorFlow must be installed manually. Either:

`$ pipenv run pip install tensorflow`

or

`$ pipenv run pip install tensorflow-gpu`

depending on whether you have a GPU. (If you run into problems, try installing TensorFlow 1.8.0,
which was used for development.)

If you want to run tests, also run:

`$ pipenv install --dev`

Finally, before running any of the scripts, enter the environment with:

`$ pipenv shell`

### Running

We support Atari environments using a feedforward policy. Basic usage is:

```
$ python3 train.py --n_workers <no. of workers> <environment name>
```

For example, to train an agent on Pong using 16 workers:

```
$ python3 train.py --n_workers 16 PongNoFrameskip-v4
```

Logs and checkpoints will be saved to a new directory in `runs`. To specify the name of the
directory, specify `--run_name`.

Run `train.py --help` to see full usage options, including tuneable hyperparameters.

Once training is completed, check out trained agent behaviour with:

```
$ python3 run_checkpoint.py <environment name> <checkpoint directory>
```

For example:

```
$ python3 run_checkpoint.py PongNoFrameskip-v4 runs/test-run_1534041413_cdc3bd9/checkpoints
```

## Special Features

OCD A3C comes with a couple of special testing features.

### Randomness checks

An annoying thing that happens with deep RL is a change which should be inconsequential apparently
breaking things because it inadvertently changes random seeding. To make it more obvious when this
might have happened, `train_test.py` runs a full end-to-end run checking that the weights are
exactly what they were after 100 updates in the previous version of the code.

### See through the eyes of the agent

Run `preprocessing_play.py` to play the game through the eyes of the agent using Gym's
[`play.py`](https://github.com/openai/gym/blob/a77b139e5875c2ab5c7ad894d0819a4e16c3f27f/gym/utils/play.py).
You'll see the full result of the preprocessing pipeline (including the frame stack,
spread out horizontally over time):

![](images/preprocessing_play.gif)

### See what the network sees

Still, what if we somehow manage to fumble something up between the preprocessing pipeline and
actually sending the frames into the policy network for inference or training?

Thanks to [`tf.Print`](https://www.tensorflow.org/api_docs/python/tf/Print), we can dump all data
going into the network then check offline that everything looks fine. This involves first running
`train.py` with the `--debug` flag and piping `stderr` to a log file:

```
$ python3 train.py PongNoFrameskip-v4 --debug 2> debug.log
```

Then, run `show_debug_data.py` on the resulting log:

```
$ python3 show_debug_data.py debug.log
```

You'll see the first five frame stacks used to choose actions on the first five steps in the
environment (with individual frames numbered - starting from 12 here because of the random number of
initial no-ops used to begin each episode):

![](images/step1.png)
![](images/step2.png)
![](images/step3.png)
![](images/step4.png)
![](images/step5.png)

Then the frame stack from the final state reached being fed into the network for value estimation:

![](images/step6.png)

Then the batch of the first five frames used for for the first training step with the corresponding
returns and actions:

![](images/train_batch.png)
![](images/returns.png)
![](images/actions.png)

Then the frame stack from the current state being fed into the network again to choose the first
action of the next set of steps:

![](images/step6_2.png)

And so on.

## Design notes

* The initial design ran each worker on a separate process, with replication of shared parameters
  done using Distributed TensorFlow. The hope was to increase speed by avoiding global interpreter
  lock, but Distributed TensorFlow seems to have a large overhead - it actually turned out to be
  faster to run the workers on threads, so that no replication is necessary, running only the
  environments in separate processes. This was inspired by OpenAI Baselines'
  [`SubprocVecEnv`](https://github.com/openai/baselines/blob/36ee5d17071424f30071bcdc72ff11b18c577529/baselines/common/vec_env/subproc_vec_env.py).
  
## Code layout

* Explain exactly what's tested

## Other Implementations
  
## Lessons learned

### Preprocessing

Preprocessing of Atari environment frames was by far the hardest thing to get right in this
implementation. Over the course of the project I ended up reimplementing preprocessing
two times before it was in a form I was confident was correct and was easily testable.

The final solution is heavily inspired by OpenAI Baselines'
[`atari_wrappers`](https://github.com/openai/baselines/blob/8c2aea2addc9f3ba36d4a0c937e6a2d09830afc7/baselines/common/atari_wrappers.py).
Taking a modular approach to preprocessing, with each stage of preprocessing being done using a
separate wrapper, makes both implementation and testing a lot easier.

If you're reproducing papers to learn about deep RL, I strongly recommend just using
`atari_wrappers`. Implementing the preprocessing from scratch did give me a much stronger appreciation
for the benefits of modular design, and it seems important to know about the tricks that are used
(episode reset on end-of-life and no-ops at the start of each episode came as particular surprises to
me), but it takes away so much time that could be spend learning about actual RL stuff instead.

### Monte Carlo vs. Temporal Difference Updates

At the start, I was thinking that Monte Carlo updates would be simpler to get working than
the 5-step temporal difference updates suggested by the paper - the bias introduced by the value bootstrap seemed like it might
complicate things. But during initial testing on Pong, though
MC updates did work when using heavy preprocessing (removing the frame and background, and setting the
ball and paddles to pure white), MC updates didn't work at all
when using the generic Atari preprocessing (just resizing and grayscaling the frame).

The ostensible explanation is that MC updates are much higher variance and therefore take longer
to converge. Maybe training was just too slow to show significant progress on the length of the runs
I was doing (~ 12 hours). But I'm hesitant to accept this as the full explanation, since MC updates
worked fine with the heavier preprocessing, and without taking too much longer.

Another possible explanation is the differences in gradient computation between MC and TD.
The pseudocode in the paper suggests simply summing gradients over each step. With MC
updates, because you're summing gradients over the many steps of an entire episode, you'll end up
taking a really large step in parameter space on every update. Updates that large just seem like a bad idea.
You could limit the size of the updates by e.g. dividing by the number of steps per episode,
but then you're going to be learning much more slowly
than with TD updates, where you're also making only small updates but you're making them much more
regularly.

It's still surprising that large MC updates work at all with the heavier preprocessing, though. Maybe
it's something to do with the sparsity of the observations, zeroing out everything but the ball
and paddles. But I never did completely figure this out.

In summary: even though MC updates might seem conceptually simpler, in practice MC updates
can work a lot worse than TD updates; and in addition to the standard bias/variance tradeoff
explanation, I suspect it might be partly that with MC updates, you're taking large steps infrequently,
whereas with TD updates, you're taking small steps more frequently.

### Gradient Clipping 

One explanation for why running multiple workers in parallel works well is that it decorrelates
updates, since each worker contributes different experience at any given moment.

(Note that Figure 3 in the paper suggests this explanation might apply less to A3C than the paper's other
algorithms. For A3C the main advantage may just be the increased rate of
experience accumulation. But let's stick with the standard story and see where it goes.)

Even if there's diversity between each update, though, this says nothing about the diversity of
experience /within/ each update (where each update consists of a batch of 5 steps from
one worker). My impression from inspecting gradients and RMSprop statistics here and there
is that this can cause problems: if each update is too homogenous, each update can point in
one direction kind of sharply, leading to large gradients on each update. Gradient clipping
therefore seems to be pretty important for A3C.

(I didn't experiment with this super thoroughly, though. Looking back through my notes, gradient norms
and certain RMSprop statistics were definitely different with and without gradient clipping, sometimes
up to a factor of 10 or so. And I do have notes of some runs failing apparently because of lack of gradient
clipping. Nonetheless, take the above story with a grain of salt.)

(Note that this is in contrast to A2C, where /each update/ consists of experience from all workers
and is therefore more diverse.)

Sadly, this is one hyperparameter
the paper doesn't list (and in fact, learning rate and gradient clipping were tuned for each game
individually). The value we use (`max_grad_norm` in [`params.py`](params.py)) of 5.0 was determined
through coarse line search over all five games.

My main takeaways from these experiences are:
* If you're implementing A3C or something similar, gradient clipping probably is something you'll have to take care of.
* If you're writing paper about an algorithm which uses gradient clipping, /please/ quote what clip value you use.

TODO learning rate, gradient clip, optimizer stuff

## Reproducing papers and the 80/20 rule

This reproduction was an interesting lesson in the 80/20 rule.

Overall, the project took about 150 hours. The breakdown was something like:
* First 30 hours: getting multiple workers basically working with Pong (Monte Carlo updates,
Pong-specific preprocessing)
* Next 50 hours: puzzling why generic preprocessing didn't work with Monte Carlo updates, and fixing
(red herring) bugs found along the way
* Next 20 hours: optimizing it to run at a competitive speed
* Next 20 hours: tuning hyperparameters
* Final 30 hours: getting it to run well on all games

I think 80% of my learning about reinforcement learning itself was in the first 30 hours. The rest of
the project feels like it was more about the tricks necessary to make deep RL work in practice.

This is not to say it felt like the rest of the project was wasted time. However, I do think I made
a mistake by starting from scratch rather than starting from an existing codebase.

TODO too many assumptions you could mess up




























































