import React, {useState} from "react";
import BikeMap from "../components/BikeMap";
import BikeStand from "../components/BikeStand";
import './homepage.css'


const HomePage = () => {

    return (
        <div className='app-container'>
            <BikeStand />
            <BikeMap />
        </div>
    )
}

export default HomePage