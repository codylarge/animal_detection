import EventCard from "./EventCard"

function EventList({ events }) {
  if (events.length === 0) return <p>No events yet.</p>
  return (
    <div>
      {events.map(event => (
        <EventCard key={event.id} event={event} />
      ))}
    </div>
  )
}

export default EventList