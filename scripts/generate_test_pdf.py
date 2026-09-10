"""Gera um PDF de teste (manual de produto fictício) com múltiplas páginas.

Usado apenas para popular data/test_docs/ com um exemplo de PDF real,
já que o parser de PDF precisa de um arquivo binário de verdade para
ser testado (não dá pra escrever um .pdf válido "na mão").
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "test_docs" / "product_manual.pdf"

PAGES = [
    (
        "Manual do Produto — Sensor de Temperatura TX-200",
        [
            "O TX-200 é um sensor de temperatura industrial com faixa de operação",
            "de -40C a 125C e precisão de +/-0.5C. Este manual cobre instalação,",
            "calibração, protocolo de comunicação e procedimentos de manutenção.",
        ],
    ),
    (
        "Instalação",
        [
            "O sensor deve ser fixado em superfície plana usando os dois parafusos",
            "M4 fornecidos no kit. A distância mínima recomendada de fontes de",
            "calor externas (motores, tubulações de vapor) é de 30 centímetros",
            "para evitar leituras distorcidas por radiação térmica.",
            "O cabo de sinal não deve ser instalado em paralelo com cabos de",
            "potência por mais de 2 metros sem blindagem, devido a interferência",
            "eletromagnética que pode introduzir ruído na leitura analógica.",
        ],
    ),
    (
        "Calibração",
        [
            "A calibração de fábrica é válida por 12 meses. Após esse período,",
            "recomenda-se recalibração usando um banho térmico de referência",
            "certificado, comparando a leitura do TX-200 em três pontos: 0C,",
            "50C e 100C. O desvio máximo aceitável em cada ponto é de 0.5C;",
            "desvios maiores indicam necessidade de substituição do sensor.",
        ],
    ),
    (
        "Protocolo de comunicação",
        [
            "O TX-200 se comunica via Modbus RTU sobre RS-485, com endereço",
            "configurável de 1 a 247 através de dip switches na base do sensor.",
            "O registrador 0x0001 contém a temperatura atual em décimos de grau",
            "Celsius, como inteiro com sinal. A taxa de atualização padrão é",
            "de 1 leitura por segundo, configurável até 10 leituras por segundo",
            "via registrador de configuração 0x0010.",
        ],
    ),
    (
        "Manutenção e solução de problemas",
        [
            "Leituras instáveis geralmente indicam mau contato no conector do",
            "cabo de sinal ou aterramento inadequado do painel de instalação.",
            "Se o sensor não responder ao protocolo Modbus, verifique primeiro",
            "a resistência de terminação da rede RS-485 (deve ser 120 ohms nas",
            "duas extremidades do barramento) antes de substituir o dispositivo.",
            "A vida útil esperada em condições normais de operação é de 5 anos.",
        ],
    ),
]


def generate_pdf() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=A4)
    width, height = A4
    margin = 2 * cm

    for title, paragraphs in PAGES:
        y = height - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, title)
        y -= 1 * cm

        c.setFont("Helvetica", 11)
        for line in paragraphs:
            c.drawString(margin, y, line)
            y -= 0.6 * cm

        c.showPage()

    c.save()
    print(f"PDF gerado em: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_pdf()
