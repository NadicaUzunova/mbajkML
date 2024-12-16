import React from 'react';
import './App.css';
import {BrowserRouter, Route, Routes} from "react-router-dom";
import HomePage from "./pages/HomePage";
import BikeStandsProvider from "./context/bikeStandsContext";

function App() {
  return (
    <BrowserRouter>
      <BikeStandsProvider>
        <Routes>
            <Route path="/" element={<HomePage />}/>
        </Routes>
      </BikeStandsProvider>
    </BrowserRouter>
  );
}

export default App;
