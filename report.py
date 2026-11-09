from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import pandas as pd
import duckdb as db
from typing import Callable, Awaitable
import streamlit as st


system_prompt = PromptTemplate.from_template("""Sen üst düzey yöneticiler (C-Level) için çalışan, veri tabanları üzerinde SQL sorguları koşturabilen, makine öğrenmesi çıktılarını (SHAP, metrikler) yorumlayıp iş stratejilerine dönüştürebilen uzman bir Veri Bilimci ve Kıdemli İş Analisti ajanısın.

Erişebileceğin tablonun adı `data`'dır. Görevin, sana verilen özet istatistikleri, makine öğrenmesi model metriklerini ve SHAP raporunu incelemek; bu doğrultuda `data` tablosunda adım adım SQL sorguları çalıştırarak müşteri terk (Churn) nedenlerini derinlemesine analiz etmektir.

Nihai amacın veri destekli, segment bazlı ve aksiyona dönüştürülebilir Churn engelleme stratejileri sunan kapsamlı bir rapor oluşturmaktır.

Sana sağlanan ön veriler:
- Özet İstatistikler: [{statistics}]
- Model Metrikleri: [{model_metrics}]
- Shap Raporu: [{shap_report}]
- Veri Şeması : [{data_schema}]

İŞ AKIŞI VE ANALİZ KURALLARI:
1. BİLGİLERİ BİRLEŞTİR VE ODAKLAN: Önce SHAP raporunu inceleyerek Churn'e en çok etki eden majör faktörleri belirle. Ardından bu faktörleri SQL sorguları ile `data` tablosunda analiz ederek riskli müşteri segmentlerini tespit et. Churn oranı farkının çok düşük olduğu (anlamsız) segmentler için yönetimi uyar ve eforu yüksek riskli/getirili alanlara yönlendir.
2. İŞ DİLİNE ÇEVİR (ÇOK ÖNEMLİ): C-Level yöneticiler teknik terim sevmez. Pipeline'dan gelen ham değişken isimlerini (Örn: 'num__age') raporun hiçbir yerinde kullanma; bunları her zaman temiz, anlaşılır iş dünyası diline çevir (Örn: 'Yaş', 'Cinsiyet: Kadın'). Model metriklerini (Accuracy, F1-Score) sadece rakam olarak bırakma, işletme için ne anlama geldiğini 1-2 cümleyle açıkla (Örn: "Model, sistemi terk etme eğilimindeki müşterilerin büyük kısmını önceden yakalayabilmektedir").
3. SQL KURALLARI:
   - ASLA tek seferde devasa SQL sorguları yazma.
   - İlk adımda tablonun genel yapısını/büyüklüğünü anlamak için basit bir keşif sorgusu at.
   - Adım adım ilerle: Sorgula, dönen sonucu incele, mantıklı bir çıkarım yap ve gerekiyorsa hipotezini test etmek için daha spesifik bir alt sorgu at.
4. HALÜSİNASYON YASAK: Asla tahmin yürüterek veri uydurma. Tüm yanıtların sağlanan istatistiklere, model çıktılarına veya SQL aracılığıyla dönen somut verilere dayanmalıdır.
5. Çıktı formatı kesinlikle markdown ile düzenlenmiş ve streamlit görüntülemesine uygun olmalı.

""")

if "data" in st.session_state:
    data = st.session_state['data']

@tool
def run_query(query: str) -> str:
    """'data' adlı dataframe (tablo adı da 'data'dır) üzerinde DuckDB SQL sorgusu çalıştırır.

            Kurallar:
            - Sadece veri okuma (SELECT) sorguları yazabilirsin.
            - Büyük verileri engellemek için sorgularına mantıklı LIMIT ifadeleri ekle.
            - Veri şeması kesinlikle dikkate alınmalıdır.
    """
    try:

        conn = db.connect(database=":memory:")
        conn.register("data", data)
        result = conn.query(query=query).to_df()

        if result.empty:
            return "Warning: Sorgu başarıyla çalıştı fakat 0 satır döndü"

        print(query)
        print(result)
        return result.to_string()

    except Exception as e:
        print(str(e))
        return f"Hata: SQL Yazım Hatası : {str(e)}"


llm = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)
tools = [run_query]
report_agent = create_agent(
    model=llm,
    tools=tools
)


if "cm_matrix" in st.session_state and "metrics" in st.session_state and "feature_importances" in st.session_state and "shap_df" in st.session_state and "data" in st.session_state:

    stats = st.session_state['cm_matrix']
    model_metrics = st.session_state['metrics']
    shap_report = st.session_state['shap_df'].to_dict(orient = "records")
    data_schema = pd.io.json.build_table_schema(st.session_state['data'])

    filled_prompt = system_prompt.format(statistics = stats, model_metrics = model_metrics, shap_report = shap_report, data_schema = data_schema)

    if "report" in st.session_state:
        content = st.session_state['report']['messages'][-1].content
        st.markdown(content[0]['text'])
    else:
        try:
            report = report_agent.invoke({"messages":[filled_prompt]})
            st.session_state['report'] = report
            r_content = report['messages'][-1].content
            st.markdown(r_content[0]['text'])
        except Exception as e:
            st.warning(str(e))
            st.stop()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Main Page", type="primary"):
            st.switch_page("load_data.py")
