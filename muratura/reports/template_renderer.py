# -*- coding: utf-8 -*-
"""
Template Renderer - Rendering dei template per relazioni di calcolo.

Supporta template HTML con placeholder {{ variabile }} e LaTeX con <<variabile>>.
"""

import os
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class TemplateRenderer:
    """
    Renderer per template HTML e LaTeX.
    """

    def __init__(self, templates_dir: str = None):
        """
        Inizializza il renderer.

        Args:
            templates_dir: Directory dei template. Default: ./templates
        """
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.templates_dir = templates_dir

    def render_html(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Renderizza un template HTML.

        Args:
            template_name: Nome del template (senza estensione)
            data: Dizionario con i dati da inserire

        Returns:
            HTML renderizzato
        """
        template_path = os.path.join(self.templates_dir, f"{template_name}.html")
        return self._render_file(template_path, data, style='html')

    def render_latex(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Renderizza un template LaTeX.

        Args:
            template_name: Nome del template (senza estensione)
            data: Dizionario con i dati da inserire

        Returns:
            LaTeX renderizzato
        """
        template_path = os.path.join(self.templates_dir, f"{template_name}.tex")
        return self._render_file(template_path, data, style='latex')

    def _render_file(self, template_path: str, data: Dict[str, Any], style: str = 'html') -> str:
        """
        Renderizza un file template.

        Args:
            template_path: Percorso del template
            data: Dati da inserire
            style: 'html' per {{ }} o 'latex' per << >>

        Returns:
            Contenuto renderizzato
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template non trovato: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        return self._render_string(template, data, style)

    def _render_string(self, template: str, data: Dict[str, Any], style: str = 'html') -> str:
        """
        Renderizza una stringa template.

        Args:
            template: Stringa template
            data: Dati da inserire
            style: 'html' per {{ }} o 'latex' per << >>

        Returns:
            Stringa renderizzata
        """
        # Aggiungi dati di default
        data = self._add_defaults(data)

        if style == 'html':
            # Pattern per {{ variabile }}
            pattern = r'\{\{\s*(\w+)\s*\}\}'
        else:
            # Pattern per <<variabile>>
            pattern = r'<<(\w+)>>'

        def replace(match):
            key = match.group(1)
            value = data.get(key, match.group(0))  # Mantieni placeholder se non trovato
            return str(value) if value is not None else ''

        return re.sub(pattern, replace, template)

    def _add_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggiunge valori di default."""
        defaults = {
            'data': datetime.now().strftime("%d/%m/%Y"),
            'anno': datetime.now().year,
            'project_name': 'Progetto Muratura',
            'committente': 'Da specificare',
            'progettista': 'Da specificare',
            'ubicazione': 'Da specificare',
            'codice_progetto': f"MUR-{datetime.now().strftime('%Y%m%d')}",
        }
        return {**defaults, **data}


class ReportBuilder:
    """
    Builder per costruire report completi da dati di analisi.
    """

    def __init__(self, project_data: Dict[str, Any] = None):
        """
        Inizializza il builder.

        Args:
            project_data: Dati del progetto
        """
        self.project_data = project_data or {}
        self.renderer = TemplateRenderer()

    def build_from_analysis(
        self,
        analysis_results: Dict[str, Any],
        seismic_params: Dict[str, Any] = None,
        frame_model: Dict[str, Any] = None,
        materials: Dict[str, Any] = None,
        output_format: str = 'html'
    ) -> str:
        """
        Costruisce un report completo dai risultati dell'analisi.

        Args:
            analysis_results: Risultati dell'analisi
            seismic_params: Parametri sismici
            frame_model: Modello telaio equivalente
            materials: Materiali utilizzati
            output_format: 'html' o 'latex'

        Returns:
            Report renderizzato
        """
        data = self._prepare_data(analysis_results, seismic_params, frame_model, materials)

        if output_format == 'latex':
            return self.renderer.render_latex('relazione_base', data)
        else:
            return self.renderer.render_html('relazione_base', data)

    def _prepare_data(
        self,
        analysis_results: Dict[str, Any],
        seismic_params: Dict[str, Any] = None,
        frame_model: Dict[str, Any] = None,
        materials: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Prepara i dati per il template."""
        seismic_params = seismic_params or {}
        frame_model = frame_model or {}
        materials = materials or {}

        # Dati di base
        data = {
            **self.project_data,
            'tipologia_edificio': self.project_data.get('building_type', 'Civile'),
            'tipo_costruzione': self.project_data.get('construction_type', 'Esistente'),
            'categoria_intervento': self.project_data.get('intervention_category', 'Miglioramento'),
            'n_piani': self.project_data.get('n_floors', 2),
            'n_piani_interrati': self.project_data.get('n_basement_floors', 0),
            'altezza_max': self.project_data.get('max_height', 6.0),
            'superficie_coperta': self.project_data.get('covered_area', 100),
        }

        # Vita nominale e classe uso
        data.update({
            'VN': self.project_data.get('VN', 50),
            'CU': self.project_data.get('CU', 1.0),
            'classe_uso': self.project_data.get('use_class', 'II'),
            'VR': self.project_data.get('VN', 50) * self.project_data.get('CU', 1.0),
        })

        # Livello conoscenza
        data.update({
            'livello_conoscenza': self.project_data.get('knowledge_level', 'LC2'),
            'FC': self.project_data.get('FC', 1.20),
        })

        # Materiali
        data['materiali_tabella'] = self._format_materials_table(materials)

        # Carichi
        data.update({
            'carichi_g1': self._format_loads(self.project_data.get('loads_g1', [])),
            'carichi_g2': self._format_loads(self.project_data.get('loads_g2', [])),
            'carichi_q': self._format_loads(self.project_data.get('loads_q', [])),
        })

        # Parametri sismici
        data.update(self._format_seismic_params(seismic_params))

        # Modello
        stats = frame_model.get('statistics', {})
        data.update({
            'n_maschi': stats.get('n_piers', 0),
            'n_fasce': stats.get('n_spandrels', 0),
            'n_nodi': stats.get('n_nodes', 0),
            'rigidezza_piano': self._format_floor_stiffness(frame_model),
        })

        # Analisi
        data.update({
            'metodi_analisi': self._format_analysis_methods(analysis_results),
            'analisi_modale': self._format_modal_analysis(analysis_results.get('modal', {})),
            'analisi_pushover': self._format_pushover(analysis_results.get('pushover', {})),
        })

        # Verifiche
        data.update({
            'verifiche_maschi': self._format_pier_checks(analysis_results.get('pier_checks', [])),
            'verifiche_fasce': self._format_spandrel_checks(analysis_results.get('spandrel_checks', [])),
            'verifiche_sld': self._format_sld_checks(analysis_results.get('sld_checks', {})),
            'IR': analysis_results.get('risk_index', 'N/D'),
            'classe_rischio': analysis_results.get('risk_class', 'N/D'),
        })

        # Meccanismi e rinforzi
        data.update({
            'meccanismi_locali': self._format_local_mechanisms(analysis_results.get('local_mechanisms', [])),
            'interventi_rinforzo': self._format_reinforcements(analysis_results.get('reinforcements', [])),
        })

        # Conclusioni
        data['conclusioni'] = self._generate_conclusions(analysis_results)

        return data

    def _format_materials_table(self, materials: Dict[str, Any]) -> str:
        """Formatta tabella materiali."""
        if not materials:
            return """
            <tr><td>Resistenza a compressione</td><td>f<sub>m</sub></td><td>2.4</td><td>MPa</td></tr>
            <tr><td>Resistenza a taglio</td><td>τ<sub>0</sub></td><td>0.060</td><td>MPa</td></tr>
            <tr><td>Modulo elastico</td><td>E</td><td>1500</td><td>MPa</td></tr>
            <tr><td>Modulo di taglio</td><td>G</td><td>500</td><td>MPa</td></tr>
            <tr><td>Peso specifico</td><td>w</td><td>18</td><td>kN/m³</td></tr>
            """

        rows = []
        for name, props in materials.items():
            rows.append(f"<tr><td colspan='4'><strong>{name}</strong></td></tr>")
            if 'fm' in props:
                rows.append(f"<tr><td>Resistenza compressione</td><td>f<sub>m</sub></td><td>{props['fm']}</td><td>MPa</td></tr>")
            if 'tau0' in props:
                rows.append(f"<tr><td>Resistenza taglio</td><td>τ<sub>0</sub></td><td>{props['tau0']}</td><td>MPa</td></tr>")
            if 'E' in props:
                rows.append(f"<tr><td>Modulo elastico</td><td>E</td><td>{props['E']}</td><td>MPa</td></tr>")
            if 'G' in props:
                rows.append(f"<tr><td>Modulo taglio</td><td>G</td><td>{props['G']}</td><td>MPa</td></tr>")
            if 'w' in props:
                rows.append(f"<tr><td>Peso specifico</td><td>w</td><td>{props['w']}</td><td>kN/m³</td></tr>")

        return '\n'.join(rows)

    def _format_loads(self, loads: list) -> str:
        """Formatta lista carichi."""
        if not loads:
            return "<p>Carichi non specificati.</p>"

        rows = ["<table>", "<tr><th>Elemento</th><th>Carico [kN/m²]</th></tr>"]
        for load in loads:
            rows.append(f"<tr><td>{load.get('name', '-')}</td><td>{load.get('value', '-')}</td></tr>")
        rows.append("</table>")
        return '\n'.join(rows)

    def _format_seismic_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Formatta parametri sismici."""
        result = {
            'cat_sottosuolo': params.get('soil_category', 'C'),
            'cat_topografia': params.get('topo_category', 'T1'),
            'SS': params.get('Ss', 1.50),
            'ST': params.get('St', 1.00),
            'S': params.get('S', 1.50),
            'q0': params.get('q0', 2.0),
            'KR': params.get('KR', 1.0),
            'q': params.get('q', 2.0),
        }

        # Tabella stati limite
        sl_rows = []
        stati_limite = params.get('stati_limite', {})
        for sl in ['SLO', 'SLD', 'SLV', 'SLC']:
            sl_data = stati_limite.get(sl, {})
            sl_rows.append(
                f"<tr><td>{sl}</td>"
                f"<td class='number'>{sl_data.get('TR', '-')}</td>"
                f"<td class='number'>{sl_data.get('ag', '-')}</td>"
                f"<td class='number'>{sl_data.get('F0', '-')}</td>"
                f"<td class='number'>{sl_data.get('Tc_star', '-')}</td></tr>"
            )
        result['parametri_sismici'] = '\n'.join(sl_rows)

        # Spettro (placeholder per immagine)
        result['spettro_immagine'] = '<p>[Inserire grafico spettro]</p>'

        return result

    def _format_floor_stiffness(self, frame_model: Dict[str, Any]) -> str:
        """Formatta rigidezza piani."""
        return "<p>Solai classificati come [rigidi/semirigidi/flessibili].</p>"

    def _format_analysis_methods(self, results: Dict[str, Any]) -> str:
        """Formatta metodi di analisi utilizzati."""
        methods = results.get('methods_used', ['POR'])
        items = []
        for m in methods:
            if m == 'POR':
                items.append("<li>Analisi POR (Pier Only Resistance)</li>")
            elif m == 'SAM':
                items.append("<li>Analisi SAM (Simplified Analysis of Masonry)</li>")
            elif m == 'FRAME':
                items.append("<li>Analisi a telaio equivalente con pushover</li>")
            elif m == 'FEM':
                items.append("<li>Analisi agli elementi finiti</li>")
            else:
                items.append(f"<li>Analisi {m}</li>")
        return f"<ul>{''.join(items)}</ul>"

    def _format_modal_analysis(self, modal: Dict[str, Any]) -> str:
        """Formatta analisi modale."""
        modes = modal.get('modes', [])
        if not modes:
            return "<p>Analisi modale non eseguita.</p>"

        rows = ["<table>",
                "<tr><th>Modo</th><th>T [s]</th><th>f [Hz]</th><th>Mx [%]</th><th>My [%]</th></tr>"]
        for mode in modes[:6]:
            rows.append(
                f"<tr><td>{mode.get('n', '-')}</td>"
                f"<td class='number'>{mode.get('T', '-'):.3f}</td>"
                f"<td class='number'>{mode.get('f', '-'):.2f}</td>"
                f"<td class='number'>{mode.get('Mx', '-'):.1f}</td>"
                f"<td class='number'>{mode.get('My', '-'):.1f}</td></tr>"
            )
        rows.append("</table>")
        return '\n'.join(rows)

    def _format_pushover(self, pushover: Dict[str, Any]) -> str:
        """Formatta analisi pushover."""
        if not pushover:
            return "<p>Analisi pushover non eseguita.</p>"

        return f"""
        <p>Analisi pushover eseguita in direzione {pushover.get('direction', 'X')}.</p>
        <ul>
            <li>Taglio massimo alla base: {pushover.get('Vmax', 'N/D')} kN</li>
            <li>Spostamento ultimo: {pushover.get('du', 'N/D')} mm</li>
            <li>Duttilità: {pushover.get('ductility', 'N/D')}</li>
        </ul>
        """

    def _format_pier_checks(self, checks: list) -> str:
        """Formatta verifiche maschi."""
        if not checks:
            return "<p>Verifiche non disponibili.</p>"

        rows = ["<table>",
                "<tr><th>ID</th><th>V<sub>Ed</sub> [kN]</th><th>V<sub>Rd</sub> [kN]</th><th>DCR</th><th>Esito</th></tr>"]

        for check in checks[:20]:  # Limita a 20 elementi
            dcr = check.get('DCR', 0)
            if dcr <= 0.8:
                dcr_class = 'dcr-ok'
                esito = 'OK'
            elif dcr <= 1.0:
                dcr_class = 'dcr-warning'
                esito = 'OK'
            else:
                dcr_class = 'dcr-fail'
                esito = 'NO'

            rows.append(
                f"<tr><td>{check.get('id', '-')}</td>"
                f"<td class='number'>{check.get('Ved', '-'):.1f}</td>"
                f"<td class='number'>{check.get('Vrd', '-'):.1f}</td>"
                f"<td class='number {dcr_class}'>{dcr:.2f}</td>"
                f"<td><strong class='{dcr_class}'>{esito}</strong></td></tr>"
            )
        rows.append("</table>")
        return '\n'.join(rows)

    def _format_spandrel_checks(self, checks: list) -> str:
        """Formatta verifiche fasce."""
        if not checks:
            return "<p>Verifiche fasce non disponibili.</p>"
        return self._format_pier_checks(checks)  # Stesso formato

    def _format_sld_checks(self, checks: Dict[str, Any]) -> str:
        """Formatta verifiche SLD."""
        max_drift = checks.get('max_drift', 0)
        limit = checks.get('limit', 0.005)
        ok = max_drift <= limit

        return f"""
        <table>
            <tr><th>Parametro</th><th>Valore</th><th>Limite</th><th>Verifica</th></tr>
            <tr>
                <td>Drift massimo interpiano</td>
                <td class='number'>{max_drift*100:.2f}%</td>
                <td class='number'>{limit*100:.2f}%</td>
                <td><strong class='{"dcr-ok" if ok else "dcr-fail"}'>{"OK" if ok else "NO"}</strong></td>
            </tr>
        </table>
        """

    def _format_local_mechanisms(self, mechanisms: list) -> str:
        """Formatta meccanismi locali."""
        if not mechanisms:
            return "<p>Analisi meccanismi locali non eseguita.</p>"

        rows = ["<table>",
                "<tr><th>Meccanismo</th><th>α</th><th>α<sub>min</sub></th><th>Verifica</th></tr>"]
        for mech in mechanisms:
            alpha = mech.get('alpha', 0)
            alpha_min = mech.get('alpha_min', 0)
            ok = alpha >= alpha_min
            rows.append(
                f"<tr><td>{mech.get('name', '-')}</td>"
                f"<td class='number'>{alpha:.3f}</td>"
                f"<td class='number'>{alpha_min:.3f}</td>"
                f"<td><strong class='{'dcr-ok' if ok else 'dcr-fail'}'>{'OK' if ok else 'NO'}</strong></td></tr>"
            )
        rows.append("</table>")
        return '\n'.join(rows)

    def _format_reinforcements(self, reinforcements: list) -> str:
        """Formatta interventi di rinforzo."""
        if not reinforcements:
            return "<p>Non sono previsti interventi di rinforzo.</p>"

        items = []
        for r in reinforcements:
            items.append(f"<li><strong>{r.get('type', '-')}</strong>: {r.get('description', '-')}</li>")
        return f"<ul>{''.join(items)}</ul>"

    def _generate_conclusions(self, results: Dict[str, Any]) -> str:
        """Genera conclusioni."""
        ir = results.get('risk_index', 0)
        risk_class = results.get('risk_class', 'N/D')

        if isinstance(ir, (int, float)):
            if ir >= 1.0:
                verdict = "soddisfa"
                box_class = "success-box"
            else:
                verdict = "non soddisfa"
                box_class = "danger-box"
        else:
            verdict = "richiede ulteriori verifiche per"
            box_class = "warning-box"

        return f"""
        <div class="{box_class}">
            <p>Sulla base delle analisi e verifiche effettuate, si conclude che l'edificio
            <strong>{verdict}</strong> i requisiti di sicurezza previsti dalla normativa vigente.</p>
            <p><strong>Indice di Rischio Sismico:</strong> {ir}</p>
            <p><strong>Classe di Rischio Sismico:</strong> {risk_class}</p>
        </div>
        """


def render_report(
    project_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    seismic_params: Dict[str, Any] = None,
    frame_model: Dict[str, Any] = None,
    materials: Dict[str, Any] = None,
    output_format: str = 'html'
) -> str:
    """
    Funzione helper per generare un report completo.

    Args:
        project_data: Dati del progetto
        analysis_results: Risultati dell'analisi
        seismic_params: Parametri sismici
        frame_model: Modello telaio equivalente
        materials: Materiali
        output_format: 'html' o 'latex'

    Returns:
        Report renderizzato
    """
    builder = ReportBuilder(project_data)
    return builder.build_from_analysis(
        analysis_results,
        seismic_params,
        frame_model,
        materials,
        output_format
    )
