# Notes on setting up a PyTorch + MLFlow research project

This repo contains a suggested directory structure / project template for research projects using PyTorch + MLFlow. It also contains notes on "best" practices which we've learned through trial and error. Suggestions are welcome!

## Project management best-practices and rules-of-thumb

### Use and contribute to nn-library

Our [nn-library](https://github.com/bonsai-neuro-ai/nn-library) repo collects some more useful tools/patterns. Use it and contribute to it so we all benefit from each others' work.

### Use AI tools thoughtfully

AI tools (e.g. GitHub Copilot, ChatGPT, Claude, Gemini, etc.) are great for boilerplate code and for generating code snippets. Use their coding features or inline autocomplete to save time and mental energy. Use their chat interfaces to brainstorm and organize concepts. But also assume that they will make mistakes and that you will need to verify their output. Here are some ways that I use AI tools in research:

1. __I use chat models to help me find keywords and concepts which I can then search online.__ The thing about doing research is that you are necessarily working on a problem that doesn't already have a solution somewhere on the internet. LLMs are trained on data from the internet. Ipso facto, they are _typically_ not going to be able to provide a solution to your big research problem. But they are also more than just keyword-based search engines. They can make some semantic connections and point you to related concepts that do exist out there. I often use this pattern of (1) write down a "brain dump" of my research problem and some intuitions or conjectures I have about it, and (2) ask the AI model to point out related concepts or keywords that I can then search for online. This is a great way to get a "second opinion" on your research problem and to find related work that you might not have thought of. But then it's critical to go and check the primary sources yourself. This also doesn't totally replace traditional literature-search tools.
2. __I enable code autocomplete.__ I use Copilot or Gemini to autocomplete code snippets. This is great for boilerplate code, but I always read the generated code and verify that it does what I want it to do. In fact, while many students are wary of letting AI code for them for fear of "not learning," I find this to be a great way to *learn* about a new library or framework. It just requires some curiosity and/or discipline: take the code that the AI generates and go through it line by line. Ask yourself if you would have done it differently. Click through to documentation for anything you don't recognize.
3. __I use AI to translate large blocks of code from one language/framework to another.__ While I'm wary of trusting large AI-generated files when generated _de novo_, I find that AI tools are great for _translating_ code from one language/framework to another. The AI tools seem to do better when transcribing from a given source. For example, have it translate a TensorFlow model into PyTorch, or a PyTorch model into JAX. As another example, I used Claude to translate [this torchvision-based SSL data augmentation pipeline](https://github.com/bonsai-neuro-ai/lightly/blob/dali/lightly/transforms/byol_transform.py) into [this DALI-based pipeline](https://github.com/bonsai-neuro-ai/lightly/blob/dali/lightly/transforms/dali_byol_transform.py). I then had it generate a test to verify that the two pipelines matched (it caught its own bug). The whole process took about an hour and it let me run some benchmarks on DALI vs torchvision speed without spending days learning DALI. (Ultimately I concluded that the differences were minor when working with large models, so it's great that I didn't have to learn DALI in depth in order to run that test).
4. __I use AI for code review.__ I use AI to help me review code. I find that it is good at catching some bugs, but not all. It is also good at pointing out places where the code could be more readable or more efficient. I'd say about 50% of the suggestions I get from it are useful and the other 50% I simply ignore. But the useful suggestions are often worth the time spent.

### Things will fail. Modularize into short scripts.

Jobs fail all the time. The other day I found my jobs crashing because of what was likely a memory leak *inside of PyTorch*. I've also had a job crash because someone else on the server started their own job without checking and we ended up competing for the same GPU resources (and yes, I've been on the other side of that too). Sometimes the power goes out or a disk fails or we just need to reboot.

The most effective way I've found to combat these kinds of errors is to break code up into minimal single-step scripts, also known as a __data pipeline__. When a script exits, it should save to disk all necessary state for the next phase of the project to run. When a script exits, the OS also gets all the resources back, so memory issues and the like are mitigated. This is why I follow the recipe:

* An analysis py script representing a single step in the pipeline. This script should:
  * be configurable from the command line with `argparse` or `jsonargparse` or equivalent.
  * have a `__main__` block which parses command line args and calls a single function to do the thing and save results to disk
* A wrapper shell script (easily adaptable to a SLURM job script AKA an SBATCH script) which:
  * (maybe) checks to see which jobs still need running (I acknowledge that this is uglier in Bash than it is in Python --- see comments below about mlflow and job management)
  * calls the analysis script with the appropriate command line args
* plotting scripts which load data from disk and plot with minimal further computation

As long as the scripts and plotter are communicating with each other via data stored to and loaded from disk, the whole process becomes quite resilient to failure and restarts.

### I've learned not to trust Python multiprocessing, at least for job management

Python in theory has its own tools like `multiprocessing` and `joblib` which could be organized into (1) a main process spawning analysis jobs and (2) subprocesses doing the work. An _advantage_ of doing things this way is that it's relatively easy to do things like query MLFlow for which jobs have already been run and launch only *new* jobs that haven't been run yet. This is nice when done in Python, but I've found through experience that it just isn't worth the headache of setting up multiprocessing in Python, especially because

- torch DataLoaders with `num_workers > 0` will then launch their own sub-sub-processes, and I've seen these crash. The suggested approach is indeed to set `num_workers > 0` but to use shell scripts to launch the jobs.
- the mere existince of a special `torch.multiprocessing` module inside of torch is a kind of warning sign: it tells us that torch tensors (on CUDA) are not natively compatible with multiprocessing.

Ultimately, I've found it a lot easier to write bug-free code when avoiding multithreading and multiprocessing, and not-writing-bugs is faster in the long run (for research code at least) than writing and rewriting and re-rewriting speedy but buggy code. Above, I suggested using shell scripts to manage and launch jobs. I admit this is clunky but it is the best balance of simplicity, customizability, and overall resilience that I've found so far.

Bonus thoughts:

- SLURM and other OS-level job schedulers are nice but then introduce even more overhead. I don't want to hire (or become) a sysadmin.
- Docker/Podman/Kubernetes are another option that has come up and which I've used semi-successfully in the past. They come with their own overhead and learning curve and are not worth the trouble in my mind.

### Use notebooks sparingly

Notebooks are great for quick prototyping and exploration, but they can quickly become unwieldy and prone to bugs. Prototyping and sanity-checking in notebooks is fine. Actually running jobs should be refactored into scripts.

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

### Why PyTorch?

Short answer: PyTorch isn't perfect, but it's easier to deal with the enemy you know...

Longer answer: I like PyTorch's relatively imperative model syntax and I think it makes models relatively "readable." I think this goes out the window when you start doing things like parameterizing model definitions and relying heavily on subclassing other models (all of which I am guilty of). It also goes out the window when you *want* to do explicit manipulations of the model graph, which might, say, [drive you to implement your own suite of tools for manipulating pytorch graphs](https://github.com/bonsai-neuro-ai/nn-library). But it is valuable to have everyone in the lab all be on the same page and using the same tools. The vast majority of things you may want to do in a research project won't hit any of these sharp edges, and if they do, we can work together to solve them (and then put the solution into our `nn_lib`)!

I also like (or at least feel comfortable with) other tools in the torch ecosystem like how `requires_grad` and `device`s work, the dataset/dataloader/optimizer/scheduler patterns, etc. I am open to someday trying out tensorflow or jax. But in my experience, a language or framework cannot be judged based only on how easy/hard certain things are to express. More important is how easy it is to debug when things go wrong, and relatedly how good the documentation and community support are. PyTorch has given me some headache-inducing bugs over the years, but it's also given me the tools and documentation and community forum to solve them. It will take a lot to get me to switch at this point.

### Why MLFlow?

This is a similar answer. There are many logging frameworks out there. I've tried tensorboard and MLFlow. I've also tried some custom homespun things. Google "ml experiment tracking" to find a dozen other tools out there. Almost all the tools out there are created by someone trying to make money, so almost all of them are marketed towards enterprise/business solutions. But research and data science demands are different from production demands. I started using MLFlow because it is at least _partially marketed_ at my use-case as a researcher. I also happened to be experimenting with LightningAI at the time, and Lightning has a built-in `MLFlowLogger` which worked fine enough. I no longer use Lightning in all projects, but I've stuck with MLFlow.

What does MLFlow do that is nice?

* not too much boilerplate to get logging up and running
* a pretty good web interface to watch and compare experiments at a glance
* the experiment / run abstractions help organizing things once you get used to it
* loading past runs natively returns pandas DataFrame objects, so the "load then plot" scripts become nice and simple

What do I wish was better about MLFlow?

* it can be slow to query when there are a lot of logged runs. Note: this is a bit better now than it used to be since mlflow deprecated the file-system-based store in favor of a SQLite store.
* because querying data is slow, logic like "if this job has already been run, exit early" is still slow when there are a lot of jobs.
  * In an ideal world, this logic would happen at the level of the wrapper batch script, especially when running batch jobs on a cluster. If it happens at the level of the python script, then what can happen is (1) queue a job, (2) wait for hours or days for it to run, then (3) it eventually launches only to find out that it doesn't need to be run. Figuring out some best-practices here is still to-do for us.
* the artifact and metrics logging are abstracted based on the assumed use-case that the mlflow tracking server will be on a different machine than the experiments. I wish there was an `mlflow-lite` API which knows that artifacts are stored in good-old-fashioned file paths. I don't want to have to "download" a file from `/tmp` to `/projects` on the same machine, but ultimately this is not a huge deal. Note we have an in-house wrapper [`nn_lib.utils.save_as_artifact`](https://github.com/bonsai-neuro-ai/nn-library/blob/main/src/nn_lib/utils/mlflow.py#L159) which illustrates this pattern for using `torch.save` to log artifacts to MLFlow.
* the latest versions of MLFlow lean even more into the enterprise / LLM use-case. Things like "traces" and the "model registry" are not something I've found useful yet (it's easier IMHO to just save the model state dict as an artifact).
