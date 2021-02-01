# Instalação
Tendo instalado o `python>=3.7` e o `jupyterlab`, o resto das dependências está definido no notebook da `AULA1`.

Entretanto, se preferir instalar pela linha de comando antes de rodar o notebook, execute as linhas abaixo.

```shell
# atualizar pip
pip install -U pip setuptools

# instalar OpenAI gym do branch master ===> IMPORTANTE!
pip install -e git+https://github.com/openai/gym.git@master#egg=gym

# Bokeh vizualization (maiores detalhes em github.com/bokeh/jupyter_bokeh)
pip install bokeh jupyter_bokeh

# jupyter lab (IPython notebooks)
pip install jupyterlab

# opcional
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @bokeh/jupyter_bokeh
```
