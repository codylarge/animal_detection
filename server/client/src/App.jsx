import { useState, useEffect } from "react"
import axios from "axios"
import EventList from "./components/EventList"

const API = "http://127.0.0.1:5000"

function App() {
  const [events, setEvents] = useState([])

  useEffect(() => {
    axios.get(`${API}/api/events`)
      .then(res => setEvents(res.data))
      .catch(err => console.error(err))
  }, [])

  return (
    <div>
      <h1>Animal Detection</h1>
      <EventList events={events} />
    </div>
  )
}

export default App