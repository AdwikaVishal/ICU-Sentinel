/**
 * Original inline illustration: a friendly female doctor in a white coat and
 * green scrubs. Using an SVG keeps the hero crisp, offline-safe and avoids
 * shipping an unlicensed stock photograph.
 */
export function DoctorIllustration() {
  return (
    <svg
      className="doctor-illustration"
      viewBox="0 0 300 340"
      role="img"
      aria-label="Illustration of a friendly female doctor in a white coat and green scrubs"
    >
      <defs>
        <linearGradient id="doctor-coat" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#edf3e9" />
        </linearGradient>
        <linearGradient id="doctor-scrubs" x1="0" y1="0" x2="0.8" y2="1">
          <stop offset="0" stopColor="#6e9874" />
          <stop offset="1" stopColor="#477c50" />
        </linearGradient>
        <linearGradient id="doctor-skin" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#f4cdb4" />
          <stop offset="1" stopColor="#e6b398" />
        </linearGradient>
        <linearGradient id="doctor-hair" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#4b3a32" />
          <stop offset="1" stopColor="#2e241f" />
        </linearGradient>
      </defs>

      {/* organic backdrop */}
      <path
        d="M49 150C45 86 92 30 158 32c57 2 102 42 100 102-2 61 26 107-21 150-46 42-108 37-153 8C38 263 53 214 49 150Z"
        fill="#d6e6cf"
        opacity="0.8"
      />
      <circle cx="237" cy="75" r="38" fill="#f6efe1" opacity="0.85" />
      <circle cx="52" cy="252" r="27" fill="#ffffff" opacity="0.55" />
      <ellipse cx="150" cy="326" rx="92" ry="12" fill="#315f3c" opacity="0.1" />

      {/* hair behind the head */}
      <path
        d="M103 132c-4-55 19-78 48-78 34 0 55 24 47 83l-4 48c-4 15-16 18-23 5l-3-57c-8-13-30-16-44-6l-4 60c-7 13-19 9-20-3Z"
        fill="url(#doctor-hair)"
      />

      {/* neck */}
      <path d="M132 172h36v30c0 9-8 16-18 16s-18-7-18-16Z" fill="#e4b096" />
      <path
        d="M132 188c6 8 12 12 18 12s13-4 18-12v14c0 9-8 16-18 16s-18-7-18-16Z"
        fill="#dca58a"
        opacity="0.55"
      />

      {/* body */}
      <path
        d="M72 340c2-55 13-96 48-116l30 14 30-14c35 20 46 61 48 116Z"
        fill="url(#doctor-coat)"
      />
      <path
        d="M117 222c9-5 22 2 33 12 11-10 24-17 33-12 8 14 12 44 12 118h-90c0-74 4-104 12-118Z"
        fill="url(#doctor-scrubs)"
      />

      {/* coat lapels */}
      <path
        d="M120 220c2 11-2 23-12 33l-26 87h-10c0-53 10-95 48-120Z"
        fill="#ffffff"
      />
      <path
        d="M180 220c-2 11 2 23 12 33l26 87h10c0-53-10-95-48-120Z"
        fill="#ffffff"
      />
      <path
        d="M119 225c8 2 16 8 27 21l-16 20-25-18Z"
        fill="#f8fbf5"
        stroke="#dbe7d5"
        strokeWidth="1.4"
      />
      <path
        d="M181 225c-8 2-16 8-27 21l16 20 25-18Z"
        fill="#f8fbf5"
        stroke="#dbe7d5"
        strokeWidth="1.4"
      />

      {/* coat seam and pocket */}
      <path
        d="M150 266v74"
        stroke="#dfe9da"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <rect x="95" y="280" width="27" height="20" rx="6" fill="#f1f6ed" />
      <rect x="101" y="286" width="15" height="3" rx="1.5" fill="#b9d0b4" />
      <rect x="101" y="292" width="10" height="3" rx="1.5" fill="#cfe0ca" />
      <circle cx="204" cy="284" r="7" fill="#e5f0de" />
      <path
        d="M204 280v8M200 284h8"
        stroke="#477c50"
        strokeWidth="1.6"
        strokeLinecap="round"
      />

      {/* head and face */}
      <ellipse cx="150" cy="128" rx="39" ry="47" fill="url(#doctor-skin)" />
      <ellipse cx="110" cy="135" rx="7" ry="10" fill="#e8b69b" />
      <ellipse cx="190" cy="135" rx="7" ry="10" fill="#e8b69b" />

      {/* fringe */}
      <path
        d="M111 116c3-28 25-44 42-42 19 2 34 13 37 35-14-11-29-15-43-9-12 5-23 10-36 16Z"
        fill="url(#doctor-hair)"
      />
      <path
        d="M184 98c8 11 9 25 6 40-4-12-9-21-18-28Z"
        fill="#3a2d27"
      />

      {/* eyes, brows, smile */}
      <path
        d="M130 127c3-3 7-3 10 0"
        stroke="#3b2f2a"
        strokeWidth="2.2"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M160 127c3-3 7-3 10 0"
        stroke="#3b2f2a"
        strokeWidth="2.2"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="135" cy="133" r="2.6" fill="#2f2925" />
      <circle cx="165" cy="133" r="2.6" fill="#2f2925" />
      <circle cx="136" cy="132" r="0.8" fill="#ffffff" opacity="0.8" />
      <circle cx="166" cy="132" r="0.8" fill="#ffffff" opacity="0.8" />
      <path
        d="M139 151c6 6 16 6 22 0"
        stroke="#a35f51"
        strokeWidth="2.4"
        strokeLinecap="round"
        fill="none"
      />
      <ellipse cx="128" cy="145" rx="5" ry="3" fill="#e39a87" opacity="0.35" />
      <ellipse cx="172" cy="145" rx="5" ry="3" fill="#e39a87" opacity="0.35" />

      {/* stethoscope */}
      <path
        d="M124 207c-10 24-11 48 2 59 13 11 33 10 43-3"
        stroke="#41674a"
        strokeWidth="4.5"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="173" cy="261" r="8" fill="#41674a" />
      <circle cx="173" cy="261" r="3.2" fill="#a9c9a3" />
      <circle cx="167" cy="206" r="3.5" fill="#41674a" />
      <circle cx="124" cy="206" r="3.5" fill="#41674a" />

      {/* floating health marks */}
      <g opacity="0.95">
        <circle cx="238" cy="150" r="17" fill="#ffffff" />
        <path
          d="M238 142v16M230 150h16"
          stroke="#477c50"
          strokeWidth="2.6"
          strokeLinecap="round"
        />
        <circle cx="58" cy="104" r="14" fill="#ffffff" />
        <path
          d="M58 98c-3-4-9-2-9 2 0 4 9 8 9 8s9-4 9-8c0-4-6-6-9-2Z"
          fill="#a9c9a3"
        />
        <circle cx="220" cy="226" r="10" fill="#f6efe1" />
        <path
          d="M216 226h8M220 222v8"
          stroke="#c5a36b"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
      </g>
    </svg>
  );
}
