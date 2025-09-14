# Hack the North Hackathon Submission

There’s one global leaderboard: Cua - Best State-of-the-Art Computer-Use Agent. Use any model setup you like (cloud or local). After projects are submitted, HUD runs the official benchmark; the top team earns a guaranteed YC partner interview (W26 batch). We’ll also feature winners on our blog and socials and kit the team out with swag.

Build a SOTA Computer-Use Agent using Cua (https://github.com/trycua/cua), the open-source infrastructure and agent framework for controlling real desktop and browser environments.
Submissions are evaluated in HUD's OSWorld-Verified benchmarking environment. The top-scoring team earns a secured interview with a Y Combinator partner for the next batch.


OS World Info: 

 
osworld task_demonstration
OSWorld is a first-of-its-kind scalable, real computer environment for multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across operating systems. It can serve as a unified environment for evaluating open-ended computer tasks that involve arbitrary apps (e.g., task examples in the above Fig). We also create a benchmark of 369 real-world computer tasks in OSWorld with reliable, reproducible setup and evaluation scripts. Note: 8 Google Drive tasks may require manual configuration or can be excluded (361 tasks) due to network dependencies.

Abstract
Autonomous agents that accomplish complex computer tasks with minimal human interventions have the potential to transform human-computer interaction, significantly enhancing accessibility and productivity. However, existing benchmarks either lack an interactive environment or are limited to environments specific to certain applications or domains, failing to reflect the diverse and complex nature of real-world computer use, thereby limiting the scope of tasks and agent scalability. To address this issue, we introduce OSWorld, the first-of-its-kind scalable, real computer environment for multimodal agents, supporting task setup, execution-based evaluation, and interactive learning across various operating systems such as Ubuntu, Windows, and macOS. OSWorld can serve as a unified, integrated computer environment for assessing open-ended computer tasks that involve arbitrary applications. Building upon OSWorld, we create a benchmark of 369 computer tasks involving real web and desktop apps in open domains, OS file I/O, and workflows spanning multiple applications (note: 8 Google Drive tasks may require manual setup or can be excluded for a 361-task evaluation). Each task example is derived from real-world computer use cases and includes a detailed initial state setup configuration and a custom execution-based evaluation script for reliable, reproducible evaluation. Extensive evaluation of state-of-the-art LLM/VLM-based agents on OSWorld reveals significant deficiencies in their ability to serve as computer assistants. While humans can accomplish over 72.36% of the tasks, the best model achieves only 12.24% success, primarily struggling with GUI grounding and operational knowledge. Comprehensive analysis using OSWorld provides valuable insights for developing multimodal generalist agents that were not possible with previous benchmarks.

OSWorld Environment Infrastructure
environment infrastructure
The OSWorld environment uses a configuration file for initializing tasks (highlighted in red), agent interaction, post-processing upon agent completion (highlighted in orange), retrieving files and information (highlighted in yellow), and executing the evaluation function (highlighted in green). The corresponding configuration items are highlighted in colors that match their respective components within the environment. Environments can run in parallel on a single host machine for learning or evaluation purposes. Headless operation is supported.

Data Statistics and Comparison
Below we present an overview of the main statistics of OSWorld, showcasing the outline and a broad spectrum of tasks. OSWorld contains a total of 369 tasks (and an additional 43 tasks on Windows for analysis).




Key statistics of OSWorld.
The "Supp. tasks" refers to the Windows-based tasks, that could only be used after activation due to copyright restrictions.
data-overview
data-composition
Distribution of task instructions in OSWorld
based on the app domains and operation types to showcase the content intuitively.

We make a comparison of OSWorld against some other different benchmarks for digital agents as presented below.
The columns indicate: whether they provide a controllable executable environment (Control. Exec. Env.), the ease of adding new tasks involving arbitrary applications in open domains (Environment Scalability), support for multimodal agent evaluation (Multimodal Support), support for and inclusion of cross-app tasks (Cross-App), capability to start tasks from an intermediate initial state (Intermediate Init. State), and the number of execution-based evaluation functions (# Exec.-based Eval. Func.).
✔️

Benchmark
We adopt state-of-the-art LLM and VLM from open-source representatives such as UI-TARS, Agent-S, Qwen, Mixtral and CogAgent, and closed-source ones from Operator, GPT, Gemini, and Claude families on OSWorld, as LLM and VLM agent baselines. We also explore methods such as the Set-of-Marks aided approach, which has been demonstrated to improve spatial capabilities for visual reasoning. We are actively updating the benchmark with new LLMs, VLMs and methods. Pull requests welcomed!


Two acceptable approaches for evaluation:

Manual Adjustment: You can manually configure these 8 tasks to complete the full 369 tasks evaluation
Exclude Tasks: You can exclude these 8 tasks and run 361 tasks instead - this is officially permitted and acceptable
Both approaches are valid for benchmark comparison and leaderboard submission.

Results
These are official results evaluated by our team under unified settings and environment. All models are tested with consistent evaluation protocols to ensure fair comparison.

For self-reported results and progress trends across different modalities, click here.

What are the differences among General model, Specialized model, and Agentic framework?
A General model is a model with broad, general-purpose capabilities. “Computer use” is one capability that can be elicited via prompting; the model itself can still perform other tasks such as dialogue and code generation. A Specialized model is trained specifically to serve as a computer-use agent; other capabilities are out of scope and are not emphasized in the corresponding reports. An Agentic framework organizes one or more General and Specialized models into a structured workflow—commonly, a GPT-family model acts as the planner while a proprietary or task-specific model serves as the grounder.
We will add new paradigms as they emerge.