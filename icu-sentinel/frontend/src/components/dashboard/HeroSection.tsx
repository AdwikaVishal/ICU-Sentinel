import { BellRing, Database, Play, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { DoctorIllustration } from "./DoctorIllustration";
import { SystemStatusCard, type SystemStatus } from "./SystemStatusCard";

export function HeroSection({ status }: { status: SystemStatus }) {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero__copy">
        <span className="eyebrow">
          <Sparkles size={13} aria-hidden="true" />
          ICU Sepsis Early Warning
        </span>
        <h1 id="hero-title">
          Intelligent <span className="text-green">Early Warning</span> for Sepsis
          in the ICU
        </h1>
        <p>
          Replay patient timelines, understand model predictions with SHAP
          explanations, and monitor risk alerts for earlier intervention.
        </p>
        <div className="hero__actions">
          <Link className="btn btn--primary" to="/replay">
            <Play size={16} aria-hidden="true" />
            Start Patient Replay
          </Link>
          <Link className="btn btn--secondary" to="/alerts">
            <BellRing size={16} aria-hidden="true" />
            View Alerts
          </Link>
        </div>
        <div className="hero__meta">
          <span>
            <ShieldCheck size={13} aria-hidden="true" />
            Research prototype
          </span>
          <span>
            <Database size={13} aria-hidden="true" />
            {status.isSynthetic === false
              ? "Configured processed dataset"
              : "Synthetic demonstration data"}
          </span>
        </div>
      </div>

      <div className="hero__visual">
        <DoctorIllustration />
      </div>

      <div className="hero__status">
        <SystemStatusCard status={status} />
      </div>
    </section>
  );
}
