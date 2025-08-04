// Test script to verify frontend can connect to backend
const axios = require('axios');

async function testApi() {
    console.log('🧪 Testing frontend to backend API connection...');
    
    const baseURL = 'http://localhost:8000';
    
    try {
        // Test 1: Health check
        console.log('\n1️⃣ Testing health endpoint...');
        const healthResponse = await axios.get(`${baseURL}/health`);
        console.log('✅ Health check:', healthResponse.status, healthResponse.data);
        
        // Test 2: Real-time price endpoint
        console.log('\n2️⃣ Testing real-time price endpoint...');
        const priceResponse = await axios.get(`${baseURL}/data/realtime/000001.SZ`);
        console.log('✅ Price check:', priceResponse.status);
        console.log('📊 Price data:', JSON.stringify(priceResponse.data, null, 2));
        
        // Test 3: CORS headers
        console.log('\n3️⃣ Checking CORS headers...');
        console.log('🔍 Response headers:', {
            'access-control-allow-origin': priceResponse.headers['access-control-allow-origin'],
            'access-control-allow-methods': priceResponse.headers['access-control-allow-methods'],
            'access-control-allow-headers': priceResponse.headers['access-control-allow-headers']
        });
        
    } catch (error) {
        console.error('❌ API Test failed:', error.message);
        if (error.response) {
            console.error('📤 Response status:', error.response.status);
            console.error('📤 Response data:', error.response.data);
        }
    }
}

testApi();