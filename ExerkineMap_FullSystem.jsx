import { useState, useEffect, useRef } from "react";

const C = {
  bg: "#04080F",
  panel: "#080E1A",
  border: "#0F1F38",
  borderHover: "#1A3560",
  cyan: "#00D4FF",
  violet: "#8B5CF6",
  green: "#10F5A0",
  orange: "#FF9900",
  red: "#FF4D6D",
  pink: "#E040FB",
  yellow: "#FFE566",
  text: "#DCF0FF",
  muted: "#4A6B8A",
  dim: "#1A2E48",
};

const LAYERS = [
  {
    id: "omics",
    title: "2.2.5 Multi-Omic State",
    short: "Multi-Omic Z_i",
    color: C.cyan,
    y: 0,
    equation: "Z_i = f_multi(G_i^RNA, G_i^ATAC, G_i^RRBS, P_i)",
    desc: "Unified cellular state embedding via VAE or GNN. Integrates RNA, chromatin, methylation, proteomics into latent Z_i.",
    tag: "VAE · GNN · Foundation Models",
  },
  {
    id: "prob",
    title: "2.2.3 Probabilistic Communication",
    short: "P_ij Signal",
    color: C.violet,
    y: 1,
    equation: "P_ij = σ(α log x_i(l) + γ log x_j(r) + η_ij)",
    desc: "Stochastic ligand-receptor interaction score. Bernoulli prior π_lr from STRING/Reactome. Enables Bayesian inference & Monte Carlo sim.",
    tag: "Bayesian · Monte Carlo · Bernoulli(π_lr)",
  },
  {
    id: "flux",
    title: "2.2.4 Dynamic Exerkine Flux",
    short: "dF/dt ODE",
    color: C.green,
    y: 2,
    equation: "dF_i/dt = -λF_i + Σ ω_ij F_j(t) + u_i(t)",
    desc: "Reaction-diffusion ODE. Degradation λ, neighborhood spread ω_ij, secretion u_i(t) conditioned on genomic/metabolic/env state.",
    tag: "ODE · Reaction-Diffusion · Temporal",
  },
  {
    id: "causal",
    title: "2.2.6 Causal Inference",
    short: "do-Calculus C_ij",
    color: C.orange,
    y: 3,
    equation: "C_ij = E[Y_i | do(F_j)] - E[Y_i]",
    desc: "Pearl do-calculus causal influence. Structural causal models, Bayesian networks, counterfactual simulation of exercise interventions.",
    tag: "SCM · do-Calculus · Counterfactual",
  },
  {
    id: "ensemble",
    title: "Layer 5: Tree Ensemble",
    short: "Ŷ Prediction",
    color: C.pink,
    y: 4,
    equation: "Y = Σ α_k f_k(Z)  [XGB + RF + LightGBM]",
    desc: "Stacked ensemble over diffused exerkine features. XGBoost sequential boosting, RF bagging, LightGBM histogram method.",
    tag: "XGBoost · RandomForest · LightGBM",
  },
  {
    id: "control",
    title: "2.2.8 Optimal Control",
    short: "LQR u*(t)",
    color: C.yellow,
    y: 5,
    equation: "min_u ∫(F^TQF + u^TRu)dt  s.t. Ḟ=AF+Bu",
    desc: "Linear-quadratic regulator over signaling dynamics. A=signaling matrix, B=intervention, u=exercise/therapy. Yields optimal prescription.",
    tag: "LQR · Control Theory · Optimal Rx",
  },
  {
    id: "generative",
    title: "2.2.7 Generative Feedback",
    short: "θ* Biomolecule",
    color: C.red,
    y: 6,
    equation: "θ* = argmax_θ U(G_ExerkineMap(θ))",
    desc: "Generative loop: identify exerkine gaps → design candidate biomolecules → simulate signaling → optimize therapeutic objective U.",
    tag: "Diffusion · Transformer · ProteinDesign",
  },
  {
    id: "organ",
    title: "2.2.9 Cross-Organ Transport",
    short: "G_organ Graph",
    color: C.cyan,
    y: 7,
    equation: "F_o(t+1) = Σ_o' W_oo' F_o'(t)",
    desc: "Organ-level graph G=(O,E). Nodes: muscle, liver, brain, immune. Edges: vascular/lymphatic. Enables muscle-brain, muscle-immune modeling.",
    tag: "Organ Graph · Transport · Multi-tissue",
  },
];

const SCORE_EQ = "ExerkineScore = (1/N) Σ P_activation(c_i)";

function MathBlock({ eq, color, small }) {
  return (
    <div style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: small ? 10 : 11,
      color: color || C.text,
      background: `${color || C.cyan}10`,
      border: `1px solid ${color || C.cyan}30`,
      borderRadius: 6,
      padding: small ? "5px 10px" : "8px 14px",
      marginTop: 6,
      letterSpacing: "0.02em",
      lineHeight: 1.6,
      wordBreak: "break-all",
    }}>
      {eq}
    </div>
  );
}

function LayerCard({ layer, isActive, onClick }) {
  const [hovered, setHovered] = useState(false);
  const active = isActive || hovered;
  return (
    <div
      onClick={() => onClick(layer.id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: active ? `${layer.color}12` : C.panel,
        border: `1px solid ${active ? layer.color + "66" : C.border}`,
        borderRadius: 12,
        padding: "14px 18px",
        cursor: "pointer",
        transition: "all 0.25s ease",
        boxShadow: active ? `0 0 20px ${layer.color}22` : "none",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 3, background: active ? layer.color : C.border, borderRadius: "3px 0 0 3px", transition: "background 0.25s" }} />
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.25em", color: active ? layer.color : C.muted, fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>
            {layer.title}
          </div>
          <div style={{ fontSize: 13, fontWeight: 700, color: active ? layer.color : C.text, fontFamily: "'Syne', sans-serif", marginBottom: 2 }}>
            {layer.short}
          </div>
          <div style={{ fontSize: 10, color: C.muted, fontFamily: "'JetBrains Mono', monospace" }}>
            {layer.tag}
          </div>
        </div>
        <div style={{ fontSize: 18, opacity: active ? 1 : 0.3, transition: "opacity 0.25s" }}>⬡</div>
      </div>
      {isActive && (
        <div style={{ marginTop: 10, animation: "fadeIn 0.3s ease" }}>
          <MathBlock eq={layer.equation} color={layer.color} />
          <div style={{ fontSize: 11, color: C.text, lineHeight: 1.7, marginTop: 10, fontFamily: "'Inter', sans-serif" }}>
            {layer.desc}
          </div>
        </div>
      )}
    </div>
  );
}

function FlowNode({ x, y, w, h, label, sublabel, color, active, small }) {
  return (
    <g>
      <rect x={x - w / 2} y={y - h / 2} width={w} height={h} rx={8}
        fill={active ? `${color}20` : "#080E1A"}
        stroke={active ? color : "#0F1F38"}
        strokeWidth={active ? 2 : 1}
        style={{ filter: active ? `drop-shadow(0 0 10px ${color}55)` : "none", transition: "all 0.3s" }}
      />
      <text x={x} y={y - (sublabel ? 6 : 1)} fontSize={small ? 10 : 12} fill={active ? color : "#DCF0FF"}
        fontFamily="'Syne', sans-serif" textAnchor="middle" fontWeight="700">{label}</text>
      {sublabel && <text x={x} y={y + 12} fontSize={9} fill="#4A6B8A"
        fontFamily="'JetBrains Mono', monospace" textAnchor="middle">{sublabel}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, dashed }) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  const ex = x2 - Math.cos(angle) * 8;
  const ey = y2 - Math.sin(angle) * 8;
  return (
    <g>
      <line x1={x1} y1={y1} x2={ex} y2={ey}
        stroke={color || C.muted} strokeWidth={1.5}
        strokeDasharray={dashed ? "4 4" : "none"} opacity={0.6} />
      <polygon
        points={`${x2},${y2} ${x2 - Math.cos(angle - 0.4) * 8},${y2 - Math.sin(angle - 0.4) * 8} ${x2 - Math.cos(angle + 0.4) * 8},${y2 - Math.sin(angle + 0.4) * 8}`}
        fill={color || C.muted} opacity={0.6} />
    </g>
  );
}

export default function ExerkineMapFull() {
  const [activeLayer, setActiveLayer] = useState("prob");
  const [tick, setTick] = useState(0);
  const [flowStep, setFlowStep] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTick(t => t + 1), 80);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setFlowStep(s => (s + 1) % 8), 900);
    return () => clearInterval(t);
  }, []);

  const toggleLayer = (id) => setActiveLayer(activeLayer === id ? null : id);

  // Flow diagram: central pipeline
  const flowNodes = [
    { id: "omics", x: 300, y: 60, label: "Multi-Omic Z_i", sub: "RNA · ATAC · Prot", color: C.cyan },
    { id: "prob", x: 300, y: 140, label: "P_ij Stochastic", sub: "Bayesian LR Score", color: C.violet },
    { id: "flux", x: 300, y: 220, label: "dF/dt ODE Flux", sub: "Reaction-Diffusion", color: C.green },
    { id: "causal", x: 300, y: 300, label: "do(F_j) Causal", sub: "SCM Inference", color: C.orange },
    { id: "ensemble", x: 300, y: 380, label: "Ŷ Ensemble", sub: "XGB · RF · LGBM", color: C.pink },
    { id: "control", x: 300, y: 460, label: "u*(t) LQR", sub: "Optimal Control", color: C.yellow },
  ];

  const sideNodes = [
    { x: 130, y: 220, label: "G_organ", sub: "Cross-Organ", color: C.cyan },
    { x: 470, y: 380, label: "θ* Gen.", sub: "Biomolecule", color: C.red },
  ];

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'Syne', sans-serif", color: C.text, padding: "20px 24px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes scanline { 0% { transform: translateY(-100%); } 100% { transform: translateY(600px); } }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #1A3560; border-radius: 2px; }
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: 1100, margin: "0 auto 24px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 16, marginBottom: 6 }}>
          <h1 style={{ fontSize: 26, fontWeight: 800, background: `linear-gradient(135deg, ${C.cyan} 0%, ${C.violet} 50%, ${C.green} 100%)`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            ExerkineMap
          </h1>
          <span style={{ fontSize: 11, color: C.muted, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.2em" }}>
            VIRTUAL PHYSIOLOGICAL SYSTEM v2.2
          </span>
        </div>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {["Digital Twin", "Causal Simulator", "Biomolecule Engine", "Precision Exercise Rx"].map(tag => (
            <span key={tag} style={{ fontSize: 10, color: C.cyan, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em", opacity: 0.7 }}>⬡ {tag}</span>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", display: "grid", gridTemplateColumns: "380px 1fr", gap: 20 }}>
        
        {/* LEFT: Layer cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.3em", color: C.muted, fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>
            PIPELINE LAYERS — CLICK TO EXPAND
          </div>
          {LAYERS.map(layer => (
            <LayerCard key={layer.id} layer={layer} isActive={activeLayer === layer.id} onClick={toggleLayer} />
          ))}
          {/* ExerkineScore */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, padding: "12px 18px", marginTop: 4 }}>
            <div style={{ fontSize: 9, letterSpacing: "0.25em", color: C.muted, fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>2.2.10 VALIDATION METRIC</div>
            <MathBlock eq={SCORE_EQ} color={C.green} small />
          </div>
        </div>

        {/* RIGHT: SVG flow diagram */}
        <div style={{ background: C.panel, borderRadius: 16, border: `1px solid ${C.border}`, padding: 16, position: "relative", overflow: "hidden" }}>
          
          {/* Scanline effect */}
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${C.cyan}44, transparent)`, animation: "scanline 4s linear infinite", pointerEvents: "none", zIndex: 1 }} />

          <div style={{ fontSize: 9, letterSpacing: "0.25em", color: C.muted, fontFamily: "'JetBrains Mono', monospace", marginBottom: 12, textAlign: "center" }}>
            SYSTEM ARCHITECTURE — LIVE SIGNAL FLOW
          </div>

          <svg width="100%" viewBox="0 0 600 560" style={{ display: "block" }}>
            
            {/* Background grid */}
            {Array.from({ length: 7 }).map((_, i) => (
              <line key={`h${i}`} x1={0} y1={80 * i + 20} x2={600} y2={80 * i + 20} stroke={C.border} strokeWidth={0.5} opacity={0.4} />
            ))}
            {Array.from({ length: 8 }).map((_, i) => (
              <line key={`v${i}`} x1={80 * i + 10} y1={0} x2={80 * i + 10} y2={560} stroke={C.border} strokeWidth={0.5} opacity={0.4} />
            ))}

            {/* Main flow arrows */}
            {flowNodes.slice(0, -1).map((node, i) => {
              const next = flowNodes[i + 1];
              return (
                <Arrow key={i} x1={node.x} y1={node.y + 28} x2={next.x} y2={next.y - 28}
                  color={flowStep === i ? node.color : C.dim} />
              );
            })}

            {/* Side connections */}
            <Arrow x1={130} y1={248} x2={220} y2={300} color={flowStep === 2 ? C.cyan : C.dim} dashed />
            <Arrow x1={380} y1={380} x2={470} y2={380} color={flowStep === 5 ? C.red : C.dim} dashed />
            <Arrow x1={470} y1={408} x2={380} y2={440} color={flowStep === 6 ? C.red : C.dim} dashed />

            {/* Feedback loop arrow */}
            <path d="M 300 488 Q 550 530 550 300 Q 550 80 350 70"
              stroke={C.yellow} strokeWidth={1.5} fill="none"
              strokeDasharray="6 4" opacity={flowStep === 7 ? 0.9 : 0.2}
              style={{ transition: "opacity 0.4s" }} />
            <text x={555} y={295} fontSize={8} fill={C.yellow} fontFamily="'JetBrains Mono', monospace" opacity={0.7}>
              feedback
            </text>

            {/* Main pipeline nodes */}
            {flowNodes.map((node, i) => (
              <FlowNode key={node.id} x={node.x} y={node.y} w={180} h={50}
                label={node.label} sublabel={node.sub} color={node.color}
                active={flowStep === i || activeLayer === node.id} />
            ))}

            {/* Side nodes */}
            <FlowNode x={130} y={220} w={120} h={46} label="G_organ" sublabel="Cross-Organ" color={C.cyan} active={flowStep === 2} small />
            <FlowNode x={470} y={380} w={120} h={46} label="θ* Generative" sublabel="Biomolecule" color={C.red} active={flowStep === 5 || flowStep === 6} small />

            {/* Animated signal dot */}
            {flowStep < flowNodes.length - 1 && (
              <circle r={5} fill={flowNodes[flowStep].color} opacity={0.9}>
                <animate attributeName="cx"
                  from={flowNodes[flowStep].x} to={flowNodes[flowStep + 1].x} dur="0.9s" fill="freeze" />
                <animate attributeName="cy"
                  from={flowNodes[flowStep].y + 30} to={flowNodes[flowStep + 1].y - 30} dur="0.9s" fill="freeze" />
                <animate attributeName="opacity" values="0;1;1;0" dur="0.9s" fill="freeze" />
              </circle>
            )}

            {/* Data source labels */}
            <text x={20} y={60} fontSize={8} fill={C.muted} fontFamily="'JetBrains Mono', monospace">MoTrPAC</text>
            <text x={20} y={75} fontSize={8} fill={C.muted} fontFamily="'JetBrains Mono', monospace">HuBMAP</text>
            <line x1={65} y1={60} x2={205} y2={60} stroke={C.dim} strokeWidth={1} strokeDasharray="3 3" />

            {/* Output label */}
            <text x={300} y={540} fontSize={9} fill={C.yellow} fontFamily="'JetBrains Mono', monospace" textAnchor="middle">
              → Personalized Exercise Prescription
            </text>
            <text x={300} y={555} fontSize={8} fill={C.muted} fontFamily="'JetBrains Mono', monospace" textAnchor="middle">
              ExerkineScore + LQR Optimal u*(t)
            </text>
          </svg>

          {/* Bottom legend */}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center", marginTop: 8, borderTop: `1px solid ${C.border}`, paddingTop: 10 }}>
            {[
              { label: "Active Signal", color: C.cyan },
              { label: "Causal Path", color: C.orange },
              { label: "Feedback Loop", color: C.yellow },
              { label: "Generative Branch", color: C.red },
            ].map(item => (
              <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 18, height: 2, background: item.color, borderRadius: 1 }} />
                <span style={{ fontSize: 9, color: C.muted, fontFamily: "'JetBrains Mono', monospace" }}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
