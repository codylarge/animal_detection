function EventCard({ event }) {
  return (
    <div>
      <h3>{event.top_species}</h3>
      <p>{event.timestamp}</p>
      <p>Total hits: {event.total_hits}</p>
    </div>
  )
}

export default EventCard