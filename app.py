import os
from flask import Flask, render_template, request
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Agent'ı ve araçları kuran fonksiyonu import et
from langChain import create_agent_executor

# Flask uygulamasını başlat
app = Flask(__name__)

# --- Agent'ı Yükle ---
# Web sunucusu başlarken agent'ı YALNIZCA BİR KEZ kuruyoruz.
# Bu, her istekte kurulum maliyetinden kaçınmamızı sağlar.
print("Web sunucusu başlıyor, lütfen bekleyin...")
try:
    agent_executor, system_prompt = create_agent_executor()
    print("🚀 Agent başarıyla yüklendi ve web sunucusu hazır.")
except Exception as e:
    print(f"❌ FATAL: Agent kurulumu sırasında bir hata oluştu: {e}")
    agent_executor = None
    system_prompt = "Agent could not be initialized."


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""

    # Eğer form gönderildiyse (kullanıcı soru sorduysa)
    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if agent_executor and question:
            print(f"🔍 Yeni soru alındı: '{question}'")

            # Agent'ı çalıştırmak için mesaj listesini oluştur
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]

            initial_state = {"messages": messages}

            print("🕵️ Agent düşünürken...")
            try:
                # Agent'ı çalıştır ve nihai durumu al
                final_state = agent_executor.invoke(initial_state, {"recursion_limit": 10})

                # Son mesajı (AI'nın cevabını) al
                final_message = final_state['messages'][-1]

                if isinstance(final_message, AIMessage):
                    answer = final_message.content
                else:
                    # Bazen sonuç farklı bir formatta olabilir
                    answer = str(final_message)

                print(f"✅ Cevap oluşturuldu: '{answer[:100]}...'")

            except Exception as e:
                print(f"❌ Agent çalışırken hata oluştu: {e}")
                answer = f"Üzgünüm, sorunuzu işlerken bir hata oluştu: {e}"
        elif not agent_executor:
            answer = "Hata: Agent düzgün bir şekilde başlatılamadığı için istek işlenemiyor. Lütfen sunucu günlüklerini kontrol edin."

    # HTML şablonunu render et ve değişkenleri (cevap, soru) şablona gönder
    return render_template('index.html', answer=answer, question=question)


if __name__ == '__main__':
    # Geliştirme sunucusunu çalıştır
    # Production ortamı için Gunicorn gibi bir WSGI sunucusu kullanın
    app.run(debug=True, port=5001)