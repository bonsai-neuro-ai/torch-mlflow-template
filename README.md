# Notes on setting up a PyTorch + MLFlow research project

This repo contains a suggested directory structure / project template for research projects using PyTorch + MLFlow. It also contains notes on "best" practices which we've learned through trial and error. Suggestions are welcome!

## Project management best-practices and rules-of-thumb

### Use and contribute to nn-library

Our [nn-library](https://github.com/bonsai-neuro-ai/nn-library) repo collects some more useful tools/patterns. Use it and contribute to it so we all benefit from each others' work.

### Why PyTorch?

Short answer: PyTorch isn't perfect, but it's easier to deal with the enemy you know...

Longer answer: I like PyTorch's relatively imperative model syntax and I think it makes models relatively "readable." I think this goes out the window when you start doing things like parameterizing model definitions and relying heavily on subclassing other models (all of which I am guilty of). It also goes out the window when you *want* to do explicit manipulations of the model graph, which might, say, [drive you to implement your own suite of tools for manipulating pytorch graphs](https://github.com/bonsai-neuro-ai/nn-library).

I also like (or at least feel comfortable with) other tools in the torch ecosystem like how `requires_grad` and `device`s work, the dataset/dataloader/optimizer/scheduler patterns, etc. I am open to someday trying out tensorflow or jax. But in my experience, a language or framework cannot be judged based only on how easy/hard certain things are to express. More important is how easy it is to debug when things go wrong, and relatedly how good the documentation and community support are. PyTorch has given me some headache-inducing bugs over the years, but it's also given me the tools and documentation and community forum to solve them. It will take a lot to get me to switch at this point.

### Why MLFlow?

This is a similar answer. There are many logging frameworks out there. I've tried tensorboard and MLFlow. I've also tried some custom homespun things. Google "ml experiment tracking" to find a dozen other tools out there. Almost all of the tools out there are created by someone trying to make money, so almost all of them are marketed towards enterprise/business solutions. But research and data science demands are different from production demands. I started using MLFlow because it is at least _partially marketed_ at my use-case as a researcher. I also happened to be experimenting with LightningAI at the time, and Lightning has a built-in `MLFlowLogger` which worked fine enough.

What does MLFlow do that is nice?

* not too much boilerplate to get logging up and running
* a pretty good web interface to watch and compare experiments at a glance
* the experiment / run abstractions help organizing things once you get used to it
* loading past runs natively returns pandas DataFrame objects, so the "load then plot" scripts become nice and simple

What do I wish was better about MLFlow?

* it can be very slow because data are stored across directories/files. One can in theory set up a database backend to speed this up, but I haven't tried yet.
* because querying data is slow, logic like "if this job has already been run, exit early" is still slow when there are a lot of jobs.
  * solution 1 is to write a python script that loads *all* prior runs (one query) and then iterates over jobs that still need running... but below I explain why I prefer to do the outer-loop over jobs in shell rather than in python
  * so, solution 2 would be to query which-jobs-need-running from bash. I haven't yet found how to do this with mlflow.
  * solution 3 would be to use mlflow pipelines. These are experimental and complicated. I still find it easier to manage my own pipelines, e.g. by running `script1.sh` followed by `script2.sh`, rather than configuring yaml files.
* the artifact and metrics logging are abstracted based on the assumed use-case that the mlflow tracking server will be on a different machine than the experiments. I wish there was an `mlflow-lite` API which knows that artifacts are stored in good-old-fashioned file paths. I don't want to have to "download" a file from `/data` to `/home` on the same machine. Not hard to get around with some wrappers, though, like [`nn_lib.utils.save_as_artifact`](https://github.com/bonsai-neuro-ai/nn-library/blob/main/src/nn_lib/utils/mlflow.py#L159)

### Things will fail. Modularize into short scripts.

Jobs fail all the time. The other day I found my jobs crashing because of what was likely a memory leak *inside of PyTorch*. I've also had a job crash because someone else on the server started their own job without checking and we ended up competing for the same GPU resources (and yes, I've been on the other side of that too). Sometimes the power goes out or we just need to reboot.

The most effective way I've found to combat these kinds of errors is to break code up into minimal single-step scripts. When a script exits, it should save to disk all necessary state for the next phase of the project to run. When a script exits, the OS also gets all the resources back, so memory issues and the like are mitigated. This is why I follow the recipe:

* An analysis py script
  * configurable from the command line with `argparse` or `jsonargparse` or equivalent.
  * a `__main__` block which parses command line args and calls a single function to do the thing and save results to disk
* A wrapper shell script
  * (maybe) checks to see which jobs still need running (I acknowledge that this is uglier in Bash than it is in Python)
  * calls the py file for a grid of params
  * manages OS resources like which GPU to run on
* plotting scripts which load data from disk and plot with minimal further computation

As long as the scripts and plotter are communicating with each other via data stored to and loaded from disk, the whole process becomes quite resilient to failure and restarts.

### I've learned not to trust Python multiprocessing for job management

Python in theory has its own tools like `multiprocessing` and `joblib` which could be organized into (1) a main process spawning analysis jobs and (2) subprocesses doing the work. An _advantage_ of doing things this way is that it's relatively easy to do things like query MLFlow for which jobs have already been run and launch only *new* jobs that haven't ben run yet. This is nice when done in Python, but I've found through experience that it just isn't worth the headache of setting up multiprocessing in Python, especially because

- torch DataLoaders with `num_workers > 0` will then launch their own sub-sub-processes, and I've seen these crash
- the mere existince of a special `torch.multiprocessing` module inside of torch is a kind of warning sign: it tells us that torch tensors (on CUDA) are not natively compatible with multiprocessing.

Ultimately, I've found it a lot easier to write bug-free code when avoiding multithreading and multiprocessing, and not-writing-bugs is faster in the long run (for research code at least) than writing and rewriting and re-rewriting speedy but buggy code. Above, I suggested using shell scripts to manage and launch jobs. I admit this is clunky but it is the best balance of simplicity, customizability, and overall resilience that I've found so far.

Bonus thoughts:

- SLURM and other OS-level job schedulers are nice but then introduce even more overhead. I don't want to hire (or become) a sysadmin.
- Docker/Podman/Kubernetes are another option that has come up and which I've used semi-successfully in the past. They come with their own overhead and learning curve and are not worth the trouble in my mind.

### Models

It's worth investing time into learning how to write custom `nn.Module`s. The minimal examples are simple, and custom modules can be a great way to refactor into more readable code. This becomes even more true if you understand how torch introspects on its modules to discover parameters and submodules.

My advice is just to not shy away from writing custom Modules. Find tutorials and experiment on your own.

### Data

Likewise, understanding a bit more about torch's `Dataset` and `DataLoader` abstractions can go a long way. Some simple best-practices:

* Also don't shy away from writing a custom `Dataset`.
* Rules for train/val/test splits:
  * use training data to fit the model
  * use validation data to tune hyperparameters, do early-stopping, etc.
  * report final performance on test data that was never touched during training or hyperparameter search. (It's an unfortunately common error that people sometimes double-dip and re-use the same validation data for early-stopping or for hyperparameter search and for final performance, but that is technically a form of dataset leakage).
* Many datasets have "train" and "val" splits but no explicit public "test" split. In these cases, best practice would be to take their "val" set and treat it as a "test" set locally, then use a `random_split` of their "train" set to get your own train/val splits.
* If using `random_split`, use RNG with a manual seed, and save the seed as a parameter of the run(s). There is little reason to vary this seed parameter.
* `DataLoader` efficiency can be a major speed bottleneck. Read the docs and learn about its parameters.
  * Setting `num_workers>0` creates background threads for loading data. Contrary to what I used to think, more workers does not mean faster dataloaders. I'm now in the habit of setting `num_workers=4` or `5`
  * Rule of thumb is to set `pin_memory=True` whenever `num_workers>0` and data will be moved to CUDA. Apparently this isn't a guaranteed speedup, but I've always seen mild gains when using it.
  * Best-practices for `batch_size` is that it should be as big as possible without getting out-of-memory errors. Find this limit for a given project by (1) instantiating your biggest model and (2) trying bigger batch sizes until it fails a forward *and* backward pass (because gradients / backwards passes take additional memory)

### The tuning playbook

Lots of other advice on speeding things up and getting the most out of your models can be found on the [Google Resarch Tuning Playbook](https://github.com/google-research/tuning_playbook).
