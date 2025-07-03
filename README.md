# desafio-CAMADA

Este projeto envolveu o desenvolvimento de uma aplicação para classificação de contas contábeis semi-automatizada com auxílio de modelos de embedding e LLMs. Em conjunto, foi desenvolvida uma [User Interface](https://desafio-camada.streamlit.app/) em streamlit para auxiliar na visualização de um possível caso de uso para uma aplicação como esta.

O elemento central deste projeto é a classe LLMAccountant definida no script [src/llm_application.py](https://github.com/vitorbborges/desafio-CAMADA/blob/main/src/llm_application.py). Esta classe lida com a geração dos embeddings e o seu armazenamento em vector_stores, além disso, define uma sequência lógica de inferência em LangGraph para alternar entre classificação por votação da maioria no KNN e classificação auxiliada por LLM em casos de baixa confiabilidade.

O notebook [notebooks/test_system.ipynb](https://github.com/vitorbborges/desafio-CAMADA/blob/main/notebooks/test_system.ipynb) apresenta uma possível maneira de avaliar o resultado do sistema e traz algumas reflexões sobre possíveis melhorias futuras.
