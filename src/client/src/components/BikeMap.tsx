import React, {FC, useContext} from 'react'
import {MapContainer, Marker, Popup, TileLayer} from "react-leaflet";
import 'leaflet/dist/leaflet.css'
import './bikemap.css'
import MarkerIcon from '../assets/bike-marker-icon.png'
//@ts-ignore
import L from 'leaflet'
import {BikeStandContext} from "../context/bikeStandsContext";

const customIcon = new L.Icon({
  iconUrl: require('../assets/bike-marker-icon.png'),
  iconSize: [40, 40], // Size of the icon
  iconAnchor: [16, 32], // Point of the icon which will correspond to marker's location
  popupAnchor: [0, -32] // Point from which the popup should open relative to the iconAnchor
});

type BikeMapProps = {
}

const BikeMap: FC<BikeMapProps> = ({}) => {
    const attribution = '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    const url = 'https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png'
    const { bikeStands, changeCurrentBikeStand } = useContext(BikeStandContext)

    const onClickOnMarker = (event: any) => {
        const { lat, lng } = event.latlng

        const found = bikeStands.find((stand) => (stand.latitude === lat && stand.longitude === lng))

        if(found)
            changeCurrentBikeStand(found)
    }

    return (
        <div>
            {/*@ts-ignore*/}
            <MapContainer center={[46.54994670178013, 15.635611927857214]} zoom={14} minZoom={5} maxZoom={30} scrollWheelZoom={true}>
                {/*@ts-ignore*/}
                <TileLayer attribution={attribution} url={url}/>

                {bikeStands.map((stand) => {
                    return (
                        <>
                            {/*@ts-ignore*/}
                            <Marker position={[stand.latitude,stand.longitude]} icon={customIcon} eventHandlers={{click: (e) => onClickOnMarker(e)}}>
                                <Popup>
                                    {stand.location}
                                </Popup>
                            </Marker>
                        </>
                    )
                })}

            </MapContainer>
        </div>
    )
}

export default BikeMap