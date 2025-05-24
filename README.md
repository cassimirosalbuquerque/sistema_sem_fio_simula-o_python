# Simulação de Sistemas de Comunicação Sem Fio Celular

Este repositório contém o código Python desenvolvido para a simulação de um sistema de comunicação móvel celular, como parte do Relatório I. O objetivo principal é analisar o desempenho da rede, especificamente a Razão Sinal-Interferência mais Ruído (SINR) e a Capacidade de Canal, utilizando o método de Monte Carlo.

## 📝 Descrição

O simulador implementa um cenário de comunicação sem fio com Múltiplos Pontos de Acesso (APs) e Equipamentos de Usuário (UEs) distribuídos em uma área de cobertura. Ele avalia como a densidade de APs (M), o número de UEs (K) e o número de canais ortogonais (N) afetam os indicadores de desempenho.

## ✨ Funcionalidades

* **Posicionamento:** Distribui APs em uma grade regular e UEs aleatoriamente na área.
* **Associação:** Associa cada UE ao AP mais próximo (maior potência recebida).
* **Alocação de Canal:** Atribui um canal disponível aleatoriamente para cada UE.
* **Cálculo de Potência:** Determina a potência recebida com base em um modelo de perda de percurso.
* **Cálculo de SINR:** Calcula o SINR para cada UE, considerando o sinal desejado, a interferência co-canal e o ruído.
* **Cálculo de Capacidade:** Estima a capacidade do canal usando a fórmula de Shannon.
* **Simulação Monte Carlo:** Executa múltiplas iterações com diferentes posicionamentos de UEs para análise estatística.
* **Análise de Cenários:** Permite simular três cenários específicos (Exercícios 9, 10 e 11), incluindo um cenário de baixo SINR.
* **Visualização:** Gera gráficos da Função de Distribuição Acumulada (CDF) para SINR e Capacidade, e visualiza snapshots da rede.

## 🛠️ Tecnologias Utilizadas

* Python 3.x
* NumPy
* Matplotlib

## ⚙️ Pré-requisitos

* Python 3 instalado
* Gerenciador de pacotes `pip`

## 🚀 Instalação e Execução

1.  **Clone o repositório:**
    ```bash
    git clone <[https://github.com/cassimirosalbuquerque/sistema_sem_fio_simulacao_python]>
    cd <sistema_sem_fio_simulacao_python>
    ```

2.  **(Opcional, mas recomendado) Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o script:**
    ```bash
    python simulacao_sem_fio.py
    ```

5.  **Interaja com o script:** O programa solicitará que você digite o número do exercício que deseja simular:
    * `9`: Análise de Cobertura (K=1)
    * `10`: Análise de Capacidade (K=13)
    * `11`: Cenário Extremo de Baixo SINR (M=36, K=5, N=1)

    Após a simulação, os gráficos correspondentes serão exibidos.

## 📈 Exemplos de Saída

*(Opcional: Você pode adicionar aqui uma ou duas imagens dos gráficos gerados pelo seu script para ilustrar o resultado)*

![Exemplo de CDF](link_para_sua_imagem_cdf.png)
![Exemplo de Snapshot](link_para_sua_imagem_snapshot.png)

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---
