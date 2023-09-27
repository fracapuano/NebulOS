<div align="center">
  <a href="https://ibb.co/gTkPrng">
    <img src="https://i.ibb.co/FXYXZfD/Nebul-OS-logo.png" alt="Nebul-OS-logo" border="0">
  </a>
</div>

# NebulOS: Fair, green AI. For Real 🌿
Welcome to the GitHub repository of the 18th ASP cycle coolest project! 🚀

With NebulOS, we push the boundaries AI adoption, focusing on how to design architectures tailored for the hardware on which they run.
During this wonderful journey, we counted on the support of the amazing people at [Nebuly](https://www.nebuly.com/) (8.3k 🌟 on GitHub), as well the guidance and help by Prof. [Barbara Caputo](linkedin.com/in/barbara-caputo-a610201a7/?originalSubdomain=it) (Politecnico di Torino, Top50 Universities world-wide), and Prof. [Stefana Maja Broadbent](https://www.linkedin.com/in/stefanabroadbent/?originalSubdomain=uk) (Politecnico di Milano, Top20 Universities world-wide).

Give us a star to show your support for the project ⭐
You can find an extended abstract of this project [here](https://sites.google.com/view/nebulos)

## Foreword 📝
### Alta Scuola Politecnica (ASP)
Alta Scuola Politecnica (more [here](https://www.asp-poli.it/)) is the **joint honors program** of Italy's best technical universities, Politecnico di Milano ([18th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)) and Politecnico di Torino ([45th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)). 
Each year, 90 students from Politecnico di Milano and 60 from Politecnico di Torino are selected from a highly competitive pool and those who succeed receive free tuition for their MSc in exchange for ~1.5 years working as **student consultants** with a partner company for an industrial project.

The project we present has been carried out with the invaluable support of folks at [Nebuly](https://www.nebuly.com/), the company behind the very well-known [`nebullvm`](https://github.com/nebuly-ai/nebuly/tree/main/optimization/nebullvm) open-source AI-acceleration library 🚀

Alongside them, we have developed a stable and reliable AI-acceleration tool that capable of designing just the right network for each specific target device. 
With this, we propose a new answer to an old Deep Learning question: how to bring large models to tiny devices. **Screw forcing a circle in a square-hole**: we feel like we are the trouble-makers here, *better to change the model from the ground up!*

## Contributions 🌟
NebulOS takes a step further by adopting actual hardware-aware metrics (such as the architectures' energy consumption 🌿) to perform the automated design of Deep Neural Architectures.

## How to Reproduce the Results 💻
1. **Clone the Repository**: `git clone https://github.com/fracapuano/NebulOS.git`
2. **Install Dependencies**: 
After having made sure you have a working version of `conda` on your machine (you can double-check running the command `conda` in your terminal), go ahead:
- Creating the environment (this code has been fully tested for Python 3.10)
```bash
conda create -n nebulosenv python=3.10 -y
```
- Activating the environment
```bash
conda activate nebulosenv
```
- Installing the (very minimal) necessary requirements
```bash
pip install -r requirements.txt
```
3. **Run the Code**: Use the provided scripts and guidelines in the repository.
To reproduce our results you can simply run the following command:
```bash
python nas.py
```

To specialize your search, you can select multiple arguments. You may select those of interest to you using Python args. To see all args available you run:

```bash
python nas.py --help
```

For instance, you can specify a search for an NVIDIA Jetson Nano device on ImageNet16-120 by running:
```bash
python nas.py --device edgegpu --dataset ImageNet16-120
```
## Live-demo ⚡
Our live demo is currently hosted as an Hugging Face space. You can find it at [spaces/fracapuano/NebulOS](https://huggingface.co/spaces/fracapuano/NebulOS)

## Next modules and roadmap
We are actively working on obtaining the next results.

- [ ] Extending this work to deal with Transformer networks in NLP.
- [ ] Bring actual AI Optimization to LLMs.

## Conclusions 🌍
We really hyped up about NebulOS because we feel it is way more than an extension; it's a revolution in the field of Green-AI. This project stays as a testament of our commitment toward truly sustainable AI, and by adopting actual hardware-aware metrics, we are making a tangible difference in the world of Deep Neural Architectures. 

Join us in this journey towards a greener future! Help us keep AI beneficial to all. This time, for real.
