export interface Stat {
  label: string;
  value: string;
  detail?: string;
}

/** A row of stat tiles: a label, a value, and an optional line of context. */
export function StatTiles({ stats, stale }: { stats: Stat[]; stale?: boolean }) {
  return (
    <dl className={`tiles${stale ? ' stale' : ''}`}>
      {stats.map((s) => (
        <div key={s.label} className="tile">
          <dt>{s.label}</dt>
          <dd className="value">{s.value}</dd>
          {s.detail ? <dd className="detail">{s.detail}</dd> : null}
        </div>
      ))}
    </dl>
  );
}
