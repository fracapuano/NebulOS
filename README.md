<div align="center">
  <a href="https://ibb.co/gTkPrng">
    <img src="https://i.ibb.co/C7RMwr0/Nebul-OSS-logo.png" alt="Nebul-OSS-logo" border="0">
  </a>
</div>

# NebulOSS: Green-AI, for Real 🌿
Welcome! 👋 
This is the GitHub repository of the 18th ASP cycle coolest project! 🚀

> Big news! A live demo of our project is available [here](link_to_live_demo).

In NebulOSS, we push forward the work presented in FreeREA, focusing on delivering an end-to-end solution for automated and energy-efficient Deep Neural Network design for Computer Vision applications. You can find an extended abstract of this project [here](link_to_extended_abstract_website).

## Foreword 📝

### Alta Scuola Politecnica (ASP)
Alta Scuola Politecnica (more [here](https://www.asp-poli.it/)) is the **joint honors program** of Italy's best technical universities, Politecnico di Milano ([18th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)) and Politecnico di Torino ([45th world-wide, QS Rankings](https://www.topuniversities.com/university-rankings/university-subject-rankings/2023/engineering-technology?&page=1)). 
Each year, 90 students from Politecnico di Milano and 60 from Politecnico di Torino are selected from a highly competitive pool and those who succeed receive free tuition for their MSc in exchange for ~1.5 years working as **student consultants** with a partner company for an industrial project.

The project we present has been carried out with the invaluable support of folks at [Nebuly](https://www.nebuly.com/) (8.3k 🌟 on GitHub), the company behind the very well-known [`nebullvum`](https://github.com/nebuly-ai/nebuly/tree/main/optimization/nebullvm) open-source AI-acceleration library 🚀

Alongside them, we have developed stable and reliable AI-acceleration tools that, in a nutshell, are capable of designing just the right network for each specific target device. With this, we propose a new answer to an old Deep Learning question: how to bring large models to tiny devices. Screw forcing a circle in a square-hole: we are trouble-makers, better to change the model from the ground up!

### FreeREA
FreeREA is a custom cell-based evolution NAS algorithm that aims to maximize model accuracy while preserving size and computational constraints typical of tiny devices. It exploits an optimized combination of training-free metrics to rank architectures during the search, without the need for model training. [Read more about FreeREA](https://arxiv.org/abs/2207.05135).

### HW-NAS-Bench
HW-NAS-Bench is the first public dataset for Hardware-aware Neural Architecture Search (HW-NAS) research. It aims to democratize HW-NAS research to non-hardware experts and make HW-NAS research more reproducible and accessible. It includes the measured/estimated hardware performance of networks on various hardware devices. [Read more about HW-NAS-Bench](https://arxiv.org/abs/2103.10584).

## Contributions 🌟
NebulOSS takes a step further by adopting actual hardware-aware metrics, as presented in HW-NAS-Bench, to perform the automated design of Deep Neural Architectures. Unlike traditional methods that use proxies like the number of parameters or flops, NebulOSS focuses on the energy consumption of the models, thoroughly measured by hardware experts.

### Relevancy of Using Actual Hardware-Aware Metrics
- **Accuracy**: Provides more accurate and realistic evaluations.
- **Efficiency**: Enables optimal accuracy-cost trade-offs.
- **Accessibility**: Democratizes HW-NAS research to non-hardware experts.

## How to Reproduce the Results 💻
1. **Clone the Repository**: `git clone https://github.com/fracapuano/NebulOSS.git`
2. **Install Dependencies**: Follow the instructions in the `requirements.txt` file.
3. **Run the Code**: Use the provided scripts and guidelines in the repository.

## Live-demo ⚡
# Daje matte caccia st'app de streamlit

## Conclusions 🌍
NebulOSS is more than an extension; it's a revolution in the field of Green-AI. By adopting actual hardware-aware metrics, we are making a tangible difference in the world of Deep Neural Architectures. Join us in this journey towards a greener future!
