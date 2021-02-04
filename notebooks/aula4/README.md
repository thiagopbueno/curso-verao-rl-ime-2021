# Instalação

Tendo instalado o `python>=3.7`, `jupyterlab` e `ipywidgets`, o resto das dependências está definido no notebook da aula 4.

Entretanto, se preferir instalar pela linha de comando antes de rodar o notebook, execute as linhas abaixo.

```zsh
# atualizar o pip
pip install -U pip setuptools

# instalar o OpenAI gym do branch master ===> IMPORTANTE!
pip install -e git+https://github.com/openai/gym.git@master#egg=gym

# Demais dependências
pip install tensorflow tensorflow-probability dm-sonnet tqdm gym-cartpole-swingup pybullet

# deve ser executado por último
pip install pyglet==1.5.11
```

## Atenção
Há um [problema](https://github.com/openai/gym/issues/2101) com o `gym==0.18.0` em que a versão da dependência `pyglet` tem
problemas com o OpenGL (pelo menos no Mac OS X Big Sur). A [solução](https://github.com/openai/gym/issues/2101#issuecomment-730513761) é forçar a instalação 
de uma versão mais nova do `pyglet` via
```zsh
pip install pyglet=1.5.11
```
Isso deve ser feito **depois da instalação das demais dependências**. Caso instale o `gym` ou outra biblioteca que dependa do `pyglet`, o Pip pode voltar a versão
do `pyglet` e desfazer o resultado do bloco de código acima.
