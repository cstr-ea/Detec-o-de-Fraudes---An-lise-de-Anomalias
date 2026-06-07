import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ============================================================
# BASE DE TREINAMENTO
#
# Em um cenário real os dados viriam de:
# - Banco de dados
# - API bancária
# - Arquivos CSV
#
# Aqui utilizamos uma pequena amostra para demonstração.
# ============================================================

dados = pd.DataFrame({

    "valor": [
        100, 50, 80, 120, 95,
        5000, 7000, 9000
    ],

    "hora": [
        14, 10, 16, 15, 13,
        2, 3, 1
    ],

    "distancia_km": [
        2, 1, 3, 2, 1,
        120, 250, 400
    ],

    "transacoes_24h": [
        3, 1, 2, 2, 1,
        15, 20, 25
    ]
})

# ============================================================
# PIPELINE
#
# StandardScaler:
# Padroniza os valores para evitar influência de escalas.
#
# IsolationForest:
# Algoritmo de detecção de anomalias.
# ============================================================

modelo = Pipeline([

    (
        "normalizacao",
        StandardScaler()
    ),

    (
        "detector",
        IsolationForest(
            contamination=0.10,
            random_state=42,
            n_estimators=200
        )
    )

])

# ============================================================
# TREINAMENTO
# ============================================================

modelo.fit(dados)

print("Modelo treinado com sucesso.\n")

# ============================================================
# FUNÇÃO PRINCIPAL
#
# Recebe uma transação e retorna:
# - Score de risco
# - Nível de risco
# - Possível fraude ou não
# ============================================================

def analisar_transacao(
        valor,
        hora,
        distancia_km,
        transacoes_24h):

    # Criação do DataFrame com a nova transação

    nova_transacao = pd.DataFrame([{

        "valor": valor,
        "hora": hora,
        "distancia_km": distancia_km,
        "transacoes_24h": transacoes_24h

    }])

    # Predição do modelo
    # -1 = anomalia
    # 1 = comportamento normal

    resultado = modelo.predict(
        nova_transacao
    )[0]

    # Score matemático do algoritmo

    score_original = modelo.decision_function(
        nova_transacao
    )[0]

    # Conversão para percentual de risco

    score_risco = round(
        (1 - ((score_original + 0.5) / 1.0))
        * 100,
        2
    )

    score_risco = max(
        0,
        min(score_risco, 100)
    )

    # ========================================================
    # REGRAS EXTRAS
    #
    # Mesmo se o modelo não detectar fraude,
    # regras de negócio podem aumentar o risco.
    # ========================================================

    risco_extra = 0

    if valor > 5000:
        risco_extra += 15

    if hora <= 4:
        risco_extra += 10

    if distancia_km > 100:
        risco_extra += 10

    if transacoes_24h > 10:
        risco_extra += 15

    score_risco += risco_extra

    score_risco = min(
        score_risco,
        100
    )

    # ========================================================
    # Classificação do risco
    # ========================================================

    if score_risco < 40:
        nivel = "BAIXO"

    elif score_risco < 70:
        nivel = "MEDIO"

    else:
        nivel = "ALTO"

    # ========================================================
    # Resultado final
    # ========================================================

    return {

        "Possivel_Fraude":
            resultado == -1,

        "Score_Risco":
            score_risco,

        "Nivel_Risco":
            nivel

    }

# ============================================================
# EXEMPLO DE TESTE
#
# Cenário suspeito:
# - Valor alto
# - Madrugada
# - Distância muito grande
# - Muitas transações em pouco tempo
# ============================================================

resultado = analisar_transacao(

    valor=8500,
    hora=2,
    distancia_km=300,
    transacoes_24h=18

)

# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================

print("===== ANALISE DE TRANSACAO =====")
print()

print(
    f"Possível Fraude: "
    f"{resultado['Possivel_Fraude']}"
)

print(
    f"Score de Risco: "
    f"{resultado['Score_Risco']}%"
)

print(
    f"Nível de Risco: "
    f"{resultado['Nivel_Risco']}"
)
