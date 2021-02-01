# Curso de Verão - IME-USP - 2021 - Introdução ao Aprendizado por Reforço

Nesse repositório colocaremos todo código necessário para os exercícios práticos de implementação do curso.


## Instalação


### Ambiente Virtual

Sugerimos a instalação dos pacotes e bibliotecas que usaremos no curso em algum ambiente virtual de sua escolha. Uma possibilidade é usar o `virtualenv` conforme instruções abaixo:

```bash
# instalar criador de ambiente virtual
# https://virtualenv.pypa.io/en/latest/installation.html
virtualenv -p python3.7 .  # criar ambiente virtual
source bin/activate        # ativar ambiente virtual
```

### Jupyter Lab

Todos os exercícios práticos e experimentos computacionais serão realizados dentro do [Jupyter Lab](https://jupyter.org/). Para instalar o pacote `jupyterlab` basta utilizar o `pip`:

```bash
pip install jupyterlab ipywidgets
```

### Outros pacotes e bibliotecas

Os requisitos de instalação para cada aula prática se encontram no respectivo arquivo README.md de cada aula. Por exemplo, para a aula 1 siga as instruções do arquivo `aula1/README.md`. Se prefirir, é possível instalar as dependências de pacotes diretamente de dentro de cada notebook (procure pela subseção `Instalação`).
