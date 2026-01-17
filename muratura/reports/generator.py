# -*- coding: utf-8 -*-
"""
Report Generator - Generazione relazioni di calcolo.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ReportGenerator:
    """
    Genera relazioni di calcolo strutturale in vari formati.
    Supporta PDF, DOCX, HTML e LaTeX.
    """

    def __init__(self, project_data: Dict[str, Any] = None):
        """
        Inizializza il generatore.

        Args:
            project_data: Dati del progetto
        """
        self.project_data = project_data or {}
        self.chapters = []
        self.figures = []
        self.tables = []

    def add_chapter(self, title: str, content: str, level: int = 1):
        """Aggiunge un capitolo alla relazione."""
        self.chapters.append({
            'title': title,
            'content': content,
            'level': level,
        })

    def add_figure(self, path: str, caption: str, label: str = None):
        """Aggiunge una figura alla relazione."""
        self.figures.append({
            'path': path,
            'caption': caption,
            'label': label or f"fig_{len(self.figures)+1}",
        })

    def add_table(self, headers: List[str], data: List[List[Any]], caption: str = ""):
        """Aggiunge una tabella alla relazione."""
        self.tables.append({
            'headers': headers,
            'data': data,
            'caption': caption,
        })

    def generate_html(self, output_path: str) -> bool:
        """
        Genera relazione in formato HTML.

        Args:
            output_path: Percorso file di output

        Returns:
            True se generazione riuscita
        """
        html = self._build_html()

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return True
        except Exception as e:
            print(f"Errore generazione HTML: {e}")
            return False

    def _build_html(self) -> str:
        """Costruisce il documento HTML."""
        project_name = self.project_data.get('name', 'Progetto Muratura')
        date = datetime.now().strftime("%d/%m/%Y")

        html = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relazione di Calcolo - {project_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            line-height: 1.6;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: #fff;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{ max-width: 100%; }}
        .figure-caption {{ font-style: italic; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RELAZIONE DI CALCOLO STRUTTURALE</h1>
        <h2>{project_name}</h2>
        <p>Analisi strutturale edificio in muratura - NTC 2018</p>
        <p>Data: {date}</p>
    </div>
"""

        # Capitoli
        for chapter in self.chapters:
            level = chapter.get('level', 1)
            tag = f"h{min(level + 1, 6)}"
            html += f"\n    <{tag}>{chapter['title']}</{tag}>\n"
            html += f"    <p>{chapter['content']}</p>\n"

        # Footer
        html += """
    <div class="footer">
        <p>Generato con MURATURA 3.0 - Software per analisi strutturale edifici in muratura</p>
        <p>Conforme a NTC 2018 e Circolare esplicativa</p>
    </div>
</body>
</html>
"""
        return html

    def generate_standard_report(self) -> None:
        """Genera una relazione standard con tutti i capitoli."""
        # 1. Premessa
        self.add_chapter(
            "Premessa",
            "La presente relazione di calcolo è redatta ai sensi delle Norme Tecniche "
            "per le Costruzioni (NTC 2018 - D.M. 17/01/2018) e della relativa Circolare "
            "esplicativa n. 7/2019."
        )

        # 2. Descrizione opera
        self.add_chapter(
            "Descrizione dell'Opera",
            f"L'edificio oggetto di analisi è un fabbricato in muratura portante.\n"
            f"Tipologia: {self.project_data.get('building_type', 'Non specificata')}\n"
            f"Categoria intervento: {self.project_data.get('intervention_category', 'Non specificata')}"
        )

        # 3. Normative
        self.add_chapter(
            "Normative di Riferimento",
            "• D.M. 17/01/2018 - Norme Tecniche per le Costruzioni\n"
            "• Circolare 21/01/2019 n. 7 - Istruzioni per l'applicazione delle NTC 2018\n"
            "• Eurocodice 8 - Progettazione delle strutture per la resistenza sismica"
        )

        # 4. Materiali
        self.add_chapter(
            "Caratteristiche dei Materiali",
            "I materiali utilizzati sono conformi alla normativa NTC 2018.\n"
            "Le proprietà meccaniche della muratura sono state determinate secondo "
            "la Tabella C8.5.I della Circolare."
        )

        # 5. Azioni
        self.add_chapter(
            "Azioni sulla Struttura",
            "Sono state considerate le seguenti azioni:\n"
            "• Carichi permanenti strutturali (G1)\n"
            "• Carichi permanenti non strutturali (G2)\n"
            "• Carichi variabili (Q)\n"
            "• Azione sismica (E)"
        )

        # 6. Combinazioni
        self.add_chapter(
            "Combinazioni di Carico",
            "Le combinazioni di carico sono state definite secondo §2.5.3 delle NTC 2018."
        )

        # 7. Modello
        self.add_chapter(
            "Modello Strutturale",
            "L'edificio è stato modellato mediante telaio equivalente, secondo le "
            "indicazioni della Circolare §C8.7.1."
        )

        # 8. Analisi
        self.add_chapter(
            "Analisi Strutturale",
            "L'analisi è stata condotta con i seguenti metodi:\n"
            "• Analisi POR - Pier Only Resistance\n"
            "• Analisi SAM - Simplified Analysis of Masonry"
        )

        # 9. Verifiche
        self.add_chapter(
            "Verifiche di Sicurezza",
            "Le verifiche sono state condotte agli stati limite ultimi (SLU) e "
            "di esercizio (SLE) secondo le prescrizioni delle NTC 2018."
        )

        # 10. Conclusioni
        self.add_chapter(
            "Conclusioni",
            "Sulla base delle analisi e verifiche effettuate, si conclude che "
            "l'edificio [soddisfa/non soddisfa] i requisiti di sicurezza previsti "
            "dalla normativa vigente."
        )
