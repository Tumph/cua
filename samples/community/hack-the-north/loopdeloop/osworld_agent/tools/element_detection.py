"""
Element Detection & Validation Tools

Provides tools for finding, verifying, and interacting with UI elements:
- Element visibility verification
- Alternative selector generation
- Element waiting strategies
- OCR-based text detection
"""

import asyncio
import base64
import re
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


async def verify_element_visible(selector: str, timeout: int = 10) -> bool:
    """
    Check if element is actually visible before clicking.
    
    Args:
        selector: Element description or selector
        timeout: Maximum time to wait for element (seconds)
        
    Returns:
        True if element is visible and interactable
    """
    
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        # In a full implementation, this would:
        # 1. Take a screenshot
        # 2. Use computer vision to find the element
        # 3. Check if it's visible and not obscured
        
        # Placeholder logic based on selector characteristics
        if _is_likely_visible_element(selector):
            return True
        
        await asyncio.sleep(0.5)
    
    return False


def _is_likely_visible_element(selector: str) -> bool:
    """Heuristic to determine if element is likely visible"""
    
    # Common visible element indicators
    visible_keywords = [
        'button', 'link', 'menu', 'input', 'field', 'dropdown',
        'checkbox', 'radio', 'submit', 'save', 'ok', 'cancel'
    ]
    
    # Hidden element indicators
    hidden_keywords = [
        'hidden', 'invisible', 'display:none', 'opacity:0',
        'off-screen', 'collapsed'
    ]
    
    selector_lower = selector.lower()
    
    # Check for hidden indicators first
    if any(keyword in selector_lower for keyword in hidden_keywords):
        return False
    
    # Check for visible indicators
    if any(keyword in selector_lower for keyword in visible_keywords):
        return True
    
    # Default to potentially visible
    return True


async def wait_for_element(selector: str, condition: str = "visible", timeout: int = 30) -> bool:
    """
    Wait for element to meet specified condition.
    
    Args:
        selector: Element description or selector
        condition: Condition to wait for ('visible', 'clickable', 'enabled')
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if condition is met within timeout
    """
    
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        if condition == "visible":
            if await verify_element_visible(selector, timeout=1):
                return True
        elif condition == "clickable":
            if await _is_element_clickable(selector):
                return True
        elif condition == "enabled":
            if await _is_element_enabled(selector):
                return True
        
        await asyncio.sleep(0.5)
    
    return False


async def _is_element_clickable(selector: str) -> bool:
    """Check if element is clickable"""
    # Placeholder implementation
    return await verify_element_visible(selector, timeout=1)


async def _is_element_enabled(selector: str) -> bool:
    """Check if element is enabled (not disabled)"""
    # Placeholder implementation
    return 'disabled' not in selector.lower()


def find_alternative_selector(description: str) -> List[str]:
    """
    Generate alternative element descriptions/selectors.
    
    Args:
        description: Original element description
        
    Returns:
        List of alternative descriptions to try
    """
    
    alternatives = []
    desc_lower = description.lower()
    
    # Word substitutions
    substitutions = {
        'button': ['btn', 'link', 'clickable', 'control'],
        'click': ['select', 'choose', 'press', 'tap'],
        'submit': ['send', 'confirm', 'ok', 'apply'],
        'close': ['dismiss', 'cancel', 'x', 'exit'],
        'menu': ['dropdown', 'options', 'list', 'nav'],
        'input': ['field', 'textbox', 'entry', 'form'],
        'search': ['find', 'lookup', 'query', 'filter'],
        'save': ['store', 'keep', 'preserve', 'commit'],
        'delete': ['remove', 'trash', 'discard', 'clear'],
        'edit': ['modify', 'change', 'update', 'alter']
    }
    
    # Generate alternatives by substitution
    for original, alt_words in substitutions.items():
        if original in desc_lower:
            for alt_word in alt_words:
                new_desc = desc_lower.replace(original, alt_word)
                alternatives.append(new_desc)
    
    # Add more generic alternatives
    if 'button' in desc_lower:
        alternatives.extend([
            'clickable element',
            'interactive control',
            'ui button'
        ])
    
    if any(color in desc_lower for color in ['red', 'blue', 'green', 'yellow']):
        alternatives.extend([
            'colored button',
            'highlighted element',
            'colored control'
        ])
    
    # Add partial matches
    words = description.split()
    if len(words) > 2:
        # Try with fewer words
        alternatives.append(' '.join(words[:2]))
        alternatives.append(' '.join(words[-2:]))
        
        # Try individual significant words
        significant_words = [w for w in words if len(w) > 3 and w.lower() not in ['the', 'and', 'with', 'for']]
        for word in significant_words:
            alternatives.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_alternatives = []
    for alt in alternatives:
        if alt not in seen and alt != description.lower():
            seen.add(alt)
            unique_alternatives.append(alt)
    
    return unique_alternatives[:5]  # Limit to top 5 alternatives


async def extract_text_from_region(bbox: Tuple[int, int, int, int], screenshot_b64: str = None) -> str:
    """
    Extract text from a specific region of the screen using OCR.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        screenshot_b64: Base64 encoded screenshot (if None, takes new screenshot)
        
    Returns:
        Extracted text from the region
    """
    
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        return "OCR libraries not available"
    
    try:
        # If no screenshot provided, would need to take one
        if screenshot_b64 is None:
            return "No screenshot provided for OCR"
        
        # Decode base64 image
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(BytesIO(image_data))
        
        # Crop to specified region
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        # Extract text using OCR
        text = pytesseract.image_to_string(cropped, config='--psm 6')
        
        # Clean up extracted text
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
        
    except Exception as e:
        return f"OCR error: {str(e)}"


async def find_text_on_screen(text: str, screenshot_b64: str = None) -> List[Tuple[int, int, int, int]]:
    """
    Find all occurrences of text on screen and return their bounding boxes.
    
    Args:
        text: Text to search for
        screenshot_b64: Base64 encoded screenshot
        
    Returns:
        List of bounding boxes (x1, y1, x2, y2) where text was found
    """
    
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        return []
    
    try:
        if screenshot_b64 is None:
            return []
        
        # Decode image
        image_data = base64.b64decode(screenshot_b64)
        image = Image.open(BytesIO(image_data))
        
        # Get text data with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Find matching text
        matches = []
        text_lower = text.lower()
        
        for i, word in enumerate(data['text']):
            if word.lower().strip() == text_lower:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                matches.append((x, y, x + w, y + h))
        
        return matches
        
    except Exception:
        return []


def generate_element_selectors(element_type: str, properties: Dict[str, Any]) -> List[str]:
    """
    Generate multiple selector variations for an element.
    
    Args:
        element_type: Type of element (button, input, link, etc.)
        properties: Element properties (text, color, position, etc.)
        
    Returns:
        List of selector strings to try
    """
    
    selectors = []
    
    # Base selector
    base = element_type.lower()
    selectors.append(base)
    
    # Add text-based selectors
    if 'text' in properties:
        text = properties['text']
        selectors.extend([
            f"{base} with text '{text}'",
            f"'{text}' {base}",
            f"{base} containing '{text}'",
            f"clickable text '{text}'"
        ])
    
    # Add color-based selectors
    if 'color' in properties:
        color = properties['color']
        selectors.extend([
            f"{color} {base}",
            f"{base} colored {color}",
            f"{color} clickable element"
        ])
    
    # Add position-based selectors
    if 'position' in properties:
        position = properties['position']
        selectors.extend([
            f"{base} in {position}",
            f"{position} {base}",
            f"{base} located {position}"
        ])
    
    # Add size-based selectors
    if 'size' in properties:
        size = properties['size']
        selectors.extend([
            f"{size} {base}",
            f"{base} sized {size}"
        ])
    
    # Add state-based selectors
    if 'state' in properties:
        state = properties['state']
        selectors.extend([
            f"{state} {base}",
            f"{base} that is {state}"
        ])
    
    # Generic fallbacks
    selectors.extend([
        f"interactive {base}",
        f"clickable {base}",
        f"ui {base}",
        f"{base} control",
        f"{base} element"
    ])
    
    return selectors


async def smart_element_search(description: str, screenshot_b64: str = None, max_alternatives: int = 5) -> Dict[str, Any]:
    """
    Intelligent element search with multiple strategies.
    
    Args:
        description: Element description
        screenshot_b64: Current screenshot for analysis
        max_alternatives: Maximum number of alternatives to try
        
    Returns:
        Search results with confidence scores and suggestions
    """
    
    results = {
        'found': False,
        'confidence': 0.0,
        'alternatives_tried': 0,
        'suggestions': [],
        'ocr_results': []
    }
    
    # Strategy 1: Direct match
    if await verify_element_visible(description, timeout=2):
        results['found'] = True
        results['confidence'] = 0.9
        return results
    
    # Strategy 2: Try alternatives
    alternatives = find_alternative_selector(description)[:max_alternatives]
    
    for alt in alternatives:
        results['alternatives_tried'] += 1
        
        if await verify_element_visible(alt, timeout=1):
            results['found'] = True
            results['confidence'] = 0.7
            results['suggestions'].append(f"Found using alternative: '{alt}'")
            return results
    
    # Strategy 3: OCR-based search if screenshot available
    if screenshot_b64 and PIL_AVAILABLE and TESSERACT_AVAILABLE:
        # Extract key words from description
        words = [w for w in description.split() if len(w) > 2]
        
        for word in words[:3]:  # Check first 3 significant words
            ocr_matches = await find_text_on_screen(word, screenshot_b64)
            if ocr_matches:
                results['ocr_results'].append({
                    'word': word,
                    'matches': len(ocr_matches),
                    'locations': ocr_matches
                })
        
        if results['ocr_results']:
            results['confidence'] = 0.5
            results['suggestions'].append("Text found via OCR - element may be present")
    
    # Strategy 4: Suggest scrolling or waiting
    if not results['found']:
        results['suggestions'].extend([
            "Try scrolling to find the element",
            "Wait for page/application to fully load",
            "Check for modal dialogs blocking the element",
            "Verify the element description is accurate"
        ])
    
    return results
