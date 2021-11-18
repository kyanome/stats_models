<h1 align="center">
  <b>Stats Models</b><br>
</h1>

A collection of Statistical Models (Stats Models) implemented in pytorch with focus on reproducibility. The aim of this project is to provide
a quick and simple working example for many of the cool stats models out there. 

### Requirements
- Python >= 3.7
- PyTorch >= 1.3
- CUDA enabled computing device

### Usage
```
$ cd stats_models
$ python main.py -c configs/<config-file-name.yaml>
```
**Config file template**
```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 784
  hidden_dims: 
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

exp_params:
  batch_size: 64  # Better to have a square number

trainer_params:
  max_epochs: 50
    .
    .
    .

```


----
<h2 align="center">
  <b>Results</b><br>
</h2>


| Model                                                                  | Paper                                            |Reconstruction | Samples |
|------------------------------------------------------------------------|--------------------------------------------------|---------------|---------|
| VAE ([Code][vae_code], [Config][vae_config])                           |[Link](https://arxiv.org/abs/1312.6114)           |    ![][2]     | ![][1]  |


<!--
### TODO
- [x] VanillaVAE
- [ ] Gaussian Mixture Model
- [ ] Latent dirichlet allocation
-->


### License
**Apache License 2.0**

| Permissions      | Limitations       | Conditions                       |
|------------------|-------------------|----------------------------------|
| ✔️ Commercial use |  ❌  Trademark use |  ⓘ License and copyright notice | 
| ✔️ Modification   |  ❌  Liability     |  ⓘ State changes                |
| ✔️ Distribution   |  ❌  Warranty      |                                  |
| ✔️ Patent use     |                   |                                  |
| ✔️ Private use    |                   |                                  |

