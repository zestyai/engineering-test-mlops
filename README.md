# Engineering Test MlOps

The MlOps role at ZestyAI covers a lot of different areas of expertise. It includes building data and evaluation pipelines, model optimization and serving, and building tools/libraries for ML Engineers.

It is not expected that you know how to do all of these things. There will be lots of opportunities to learn the skills that you don't have, and there will be new challenges that are unknown to us now that you will need to adapt to solve.

The purpose of this test is to demonstrate the types of problems that you might see at ZestyAI. These questions were carefully crafted to try to represent as closely as possible "real" work that you would do here. If you enjoy the work in this test, it is likely you will enjoy the work you do at ZestyAI.

## Instructions

There are several questions here. You don't need to do them all. You should pick ***two questions*** that interest you the most or that you think you can use to most demonstrate your abilities. We tried to make the instructions as specific as possible so you know what is expected, but feel free to be creative with your solutions if you have other ideas.

If you are uncertain about your answers on some, you can do more than two questions - there will be partial credit if you leave a good explanation of where you got stuck.

### Artifacts

All the questions will use the same set of artifacts (for inputs, example outputs, model checkpoints, etc.). These can be found in the [artifacts](artifacts/) directory.

The model used in all questions is a pytorch, computer vision, classification model. An example of how to create the model and load the checkpoint can also be found at [artifacts/model.py](artifacts/model.py).

We use [dvc](https://dvc.org/doc/use-cases/versioning-data-and-models) to store large files. You can simply install dvc with `pip install dvc[gs]` or refer to [DVC's installation documentation](https://dvc.org/doc/install). You can download the artifacts with `dvc pull`.

### Submission

Package your code and any relevant artifacts and submit it to the link provided to you via email.

Include a `README.md` file that provides the following things for each question:
- A brief overview of your solution.
- Step-by-step instructions for reproducing your results.
- Clear setup instructions, including any dependencies and commands required to run your code.
- Add any comments or things you want the reviewer to consider when looking at your submission.

***Important***: Ensure all instructions and any scripts included allow reviewers to easily follow and reproduce your solution without additional guidance.

## Questions

In case you missed it - *YOU ONLY NEED TO DO **TWO** QUESTIONS*.

### Model Serving
At ZestyAI, we use [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) for serving models. Triton Inference Server has multiple model backends for serving different types of models. Currently, we are mostly using the [Pytorch](https://github.com/triton-inference-server/pytorch_backend) and [Python](https://github.com/triton-inference-server/python_backend) backends.

Assuming Triton is new to you, this task is help you get familiar with this tool, and demonstrate your ability to understand very basic usage.

**Requirements:**
1. Take the [example model](./artifacts/model.py) and compile it to TorchScript.
2. Try to serve this compiled model in Triton using the pytorch backend. Try to call it as a client to make sure it works.
3. Create another python model that does preprocessing - this should do at least the resizing needed for this model, but you can demo any other preprocessing steps you think might be helpful.
4. Create a third, ensemble model, that runs preprocessing and then the model.

**Deliverables:**
- The model store for triton packaged up as zip, or python code which can generate your store from the source files (bonus points for the latter).
- Instructions on how to run Triton with your store. A bash script would be just fine as well.
- A notebook which demonstrates calling your models in Triton. You can use Triton's client or just use the Triton HTTP interface.

***HINT** Triton has lots of examples out there, many of which cover most of the things being asked here*

### Model Optimization
In the platform team, we are always looking for ways to make models run faster.  There a lots of known ways to make models faster, such as quantization, but actually implementing such an optimization has lots of "gotchas" when you try it.  Besides just getting the optimization to work correctly, there's challenges in verifying the actual performance improvements, verifying results, how to serve the model, etc. This question attempts to simulate these challenges with a small toy example.

For this question, you will be working with the [example model in the artifacts folder](artifacts/model.py). Your goal is to try different optimization techniques to make the model smaller and/or faster, benchmark the different techniques, and present your finding.  It is not expected that you actually find any good optimizations, but rather that you can work with the process, handle the challenges that come up in different techniques, and be able to present clearly the results of your experiment.

Here are a few different optimization techniques to try (but you are not limited to these, if you have other ideas):
- Quantization
- Automatic Mixed Precision
- Compiling to TensorRT
- Compiling to ONNX

*How to benchmark performance/latency*
- If you also do the previous question, `Model Serving`, you can, and should, run the benchmarks in Triton, using the [perf-analyzer](https://github.com/triton-inference-server/perf_analyzer) tool.
- If you do not do the previous question, any kind of quick and dirty benchmark will suffice.

**Deliverables:**
- The code used to generate the optimized version(s) of the model
- Any built artifacts or instructions on how to build them
- Benchmark results and other stats you have to present.
   - The clarity of your presentation of the results will be a large part of how your submission will be evaluated.
   - We don't expect a lengthy or fancy presentation. just that it's very easy to interpret the results quickly, and that conclusions are spelled out.

### Evaluation Pipeline
A big part of the work that we call MlOps involves building pipelines. What we call a "pipeline" is a series of steps that be triggered via automation. Usually each individual step in a pipeline is pretty simple - little or no code is needed. The art of pipeline building is designing the workflow and how things get broken up into parts. The "how" of pipeline building is often just about understanding the orchestration tool and putting together some configuration files.

There are lots of tools out there to orchestrate pipelines, many designed specifically for ML. At ZestyAI, we mostly use Github Actions, but we also use prefect and a few others. And the pipelines we have built specifically for ML are pretty limited. So this is an area that you would explore and develop at your role here. This would likely require experimenting with tools you are not familiar with and being able to quickly build an MVP with a new tool to try it out.

For this question, you will build a basic model evaluation pipeline, using the orchestration tool of your choice. Ideally, you will use a tool that you have limited experience with, but that's not required. The goal of this question is to demonstrate being able to think in pipelines and prototype ideas.

**Requirements**
- The pipeline should be able to easily run by the reviewers. So keep that in mind when you're choosing a tool.
- The inputs to test will be in [artifacts/inputs.csv](artifacts/inputs.csv). The "inputs" are just image paths, which will be referring images in the artifacts folder as well.
- You will be comparing to outputs at [artifacts/outputs.csv](artifacts/outputs.csv)
- Use the same [example model](./artifacts/model.py) as the other questions
    - *don't forget to preprocess (e.g. resize) the images before calling the model*
    - The example model is a classification model, with example class names listed in the model.py.
- For evaluation metrics, just show whatever you think is appropriate for this model.
    - *We are not really so concerned with what metrics you calculate - this task is mainly about designing the pipeline and not the details of each step.*
    - As a bonus, you can allow custom metric functions as an input to your pipeline. This is a technique use sometimes currently.
- You don't need to spend too much time generalizing this pipeline.  But designing interfaces is an important part of the job, so be sure to add comments around where you might generalize or what the interface would be.

**Deliverables:**
- Your pipeline configuration files.
- Any code or additional artifacts that are part of the pipeline.
- Clear instructions on how to setup and run the pipeline.

### Model Conversion
At ZestyAI, we have lots of different Computer Vision models that classify or segment images. Often, the model is built from an initial copy that comes from an open source version of a model architecture. In the MlOps role, it is often required that you are able to dig into the code of an open source model used by an ML Engineer and find optimizations or make fixes (e.g. to make a model TorchScript-able).

For this question, you will be given a set of model weights that were the result of training a `vit` model using the model found in the open source `timm` repo. The task is to try to use the HuggingFace version of this model as a replacement. You will need to dig into the code/structure of both the timm and HuggingFace version of this model to see where the differences are and what changes need to be made to the model weights to make them work with the HuggingFace version of the model.

You should make a notebook that creates both the timm and HuggingFace versions of the model, load the original/modified versions of the weights, and run the same input on both models and verify that they both come out with the same results.

Here is the code that creates the `timm` version of the model:
```python
import timm
model1 = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=7)
sd1 = torch.load('artifacts/vit-7class.pth')
model1.load_state_dict(sd1)
```

You can use any image for testing, but there are some example images in the [artifacts](artifacts/) folder.

***NOTE** The input image will need to be resized to the expected input size for the model*

**Deliverables:**
- A notebook that does the model conversion and test to ensure results are the same.

## Thanks

We appreciate that these tests are a lot of work.  We hope you found the questions interesting and challenging. Thank you, for your time investment.

This test is always a work in progress. If something is not clear, or you have any questions/suggestions, please reach out at eng-homework@zesty.ai.
