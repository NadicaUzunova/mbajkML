import Axios from 'axios';
import dotenv from 'dotenv'

export const mbajkApi = Axios.create({
    baseURL: process.env.REACT_APP_BACKEND_API_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
    },
});