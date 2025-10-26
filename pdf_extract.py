import fitz
import pdfplumber
import pytesseract
import io
import ollama
import streamlit as st
from PIL import Image
import tempfile
import contextlib
import os

# Uyarı mesajlarını gizleme
fitz.TOOLS.mupdf_warnings()

def suppress_stdout(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            return func(*args, **kwargs)
    return wrapper

@suppress_stdout
def extract_ordered_elements(pdf_path):
    """PDF'ten öğeleri orijinal düzeninde çıkarır"""
    doc = fitz.open(pdf_path)
    elements = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")
        
        # Sayfa elementlerini işleme
        page_elements = []
        
        # Metin bloklarını ekleme
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                text = "\n".join([line["spans"][0]["text"] for line in block["lines"]])
                page_elements.append({
                    "type": "text",
                    "content": text,
                    "bbox": block["bbox"],
                    "page": page_num + 1
                })
            elif block["type"] == 1:  # Image block
                try:
                    # Farklı görsel formatları için esnek yaklaşım
                    if "xref" in block:
                        xref = block["xref"]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                    else:
                        # Alternatif görsel çıkarma yöntemi
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        image_bytes = pix.tobytes("ppm")
                    
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    page_elements.append({
                        "type": "image",
                        "content": image,
                        "bbox": block["bbox"],
                        "page": page_num + 1
                    })
                except Exception as e:
                    print(f"Sayfa {page_num + 1} görsel işlenirken hata: {str(e)}")
                    continue
        
        # Tabloları ekleme (pdfplumber ile)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                plumber_page = pdf.pages[page_num]
                tables = plumber_page.extract_tables()
                for table_num, table in enumerate(tables):
                    page_elements.append({
                        "type": "table",
                        "content": table,
                        "bbox": None,
                        "page": page_num + 1,
                        "table_num": table_num + 1
                    })
        except Exception as e:
            print(f"Sayfa {page_num + 1} tablo işlenirken hata: {str(e)}")
        
        # Elementleri Y pozisyonuna göre sırala
        page_elements.sort(key=lambda x: x["bbox"][1] if x["bbox"] else 0)
        elements.extend(page_elements)
    
    return elements

def extract_text_from_images(elements):
    """Görsellerden OCR ile metin çıkarır"""
    for element in elements:
        if element["type"] == "image":
            try:
                text = pytesseract.image_to_string(element["content"])
                element["ocr_text"] = text.strip()
            except Exception as e:
                print(f"OCR işlenirken hata: {str(e)}")
                element["ocr_text"] = "OCR işlemi başarısız oldu"

def analyze_elements_with_ollama(elements):
    """Elementleri LLM ile analiz eder"""
    context = "PDF İçeriği:\n\n"
    for idx, element in enumerate(elements, 1):
        context += f"\n\nÖğe {idx} (Sayfa {element['page']}):\n"
        if element["type"] == "text":
            context += f"Metin:\n{element['content']}"
        elif element["type"] == "image":
            context += f"Görsel - OCR Metni:\n{element.get('ocr_text', 'Metin bulunamadı')}"
        elif element["type"] == "table":
            context += "Tablo İçeriği:\n"
            for row in element["content"]:
                context += " | ".join(str(cell) if cell is not None else "" for cell in row) + "\n"
    
    prompt = "Aşağıdaki PDF içeriğini detaylıca analiz et ve özetle:\n\n" + context
    try:
        response = ollama.generate(model="gemma3", prompt=prompt)
        return response['response']
    except Exception as e:
        return f"AI analizi sırasında hata oluştu: {str(e)}"

def main():
    st.title("PDF İçerik Analiz Aracı")
    
    uploaded_file = st.file_uploader("PDF Dosyası Yükle", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            elements = extract_ordered_elements(tmp_path)
            extract_text_from_images(elements)
            
            # Önizleme Bölümü
            with st.expander("Ham İçerik Önizleme"):
                for element in elements:
                    st.subheader(f"Sayfa {element['page']} - {element['type'].capitalize()}")
                    if element["type"] == "text":
                        st.text(element["content"])
                    elif element["type"] == "image":
                        st.image(element["content"], caption=f"Görsel - Sayfa {element['page']}")
                        if "ocr_text" in element:
                            st.write("OCR Metni:")
                            st.code(element["ocr_text"])
                    elif element["type"] == "table":
                        st.table(element["content"])
            
            # Analiz Bölümü
            if st.button("Analiz Başlat"):
                with st.spinner("AI Analizi Yapılıyor..."):
                    analysis = analyze_elements_with_ollama(elements)
                    st.subheader("AI Analizi")
                    st.markdown(analysis)
        
        except Exception as e:
            st.error(f"PDF işlenirken hata oluştu: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

if __name__ == "__main__":
    main()