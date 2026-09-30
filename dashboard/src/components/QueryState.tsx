import type { UseQueryResult } from '@tanstack/react-query';
import type { ReactNode } from 'react';

/** A query's content, with its loading and error states, and the previous result dimmed while
 *  a new one loads. */
export function QueryState<T>({ query, what, children }: {
  query: UseQueryResult<T, Error>;
  what: string;
  children: (data: T) => ReactNode;
}) {
  if (query.data !== undefined) {
    return (
      <div className={query.isPlaceholderData || (query.isFetching && query.isStale) ? 'stale' : undefined}
           aria-busy={query.isFetching}>
        {query.error ? <p className="error" role="alert">{query.error.message}</p> : null}
        {children(query.data)}
      </div>
    );
  }
  if (query.error) return <p className="error" role="alert">{query.error.message}</p>;
  if (query.isFetching) return <p className="loading">Loading {what}…</p>;
  return null;
}
